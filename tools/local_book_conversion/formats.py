"""Native PDF and fixed-layout EPUB extraction, with illustration-only OCR."""

import json, re, statistics, subprocess, xml.etree.ElementTree as ET, zipfile, posixpath, io, hashlib
from urllib.parse import unquote, urlsplit
from pathlib import Path
import pypdfium2 as pdfium
import pypdfium2.raw as raw
import numpy as np
import cv2
from PIL import Image
from .common import clean, esc, norm, Timings, image_text_sections

NS = {"x": "http://www.w3.org/1999/xhtml"}


def write_image(dest, image):
    if not cv2.imwrite(str(dest), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError("Could not write image")


def native_block_order(blocks):
    """Whitespace cuts on whole native blocks; never interleave their lines."""
    lines = [l for b in blocks for l in b]
    vertical = sum(
        len(l["text"])
        for l in lines
        if l["bbox_points"][3] - l["bbox_points"][1]
        > l["bbox_points"][2] - l["bbox_points"][0]
    )
    rotated = vertical > sum(len(l["text"]) for l in lines) * 0.65
    heights = [
        min(
            l["bbox_points"][3] - l["bbox_points"][1],
            l["bbox_points"][2] - l["bbox_points"][0],
        )
        for l in lines
        if l["text"]
    ]
    size = statistics.median(heights) if heights else 10

    def bbox(block):
        x0 = min(l["bbox_points"][0] for l in block)
        y0 = min(l["bbox_points"][1] for l in block)
        x1 = max(l["bbox_points"][2] for l in block)
        y1 = max(l["bbox_points"][3] for l in block)
        return [-y1, x0, -y0, x1] if rotated else [x0, y0, x1, y1]

    def walk(items):
        if len(items) < 2:
            return items
        cuts = []
        for axis, threshold in [(0, size * 0.7), (1, size * 0.5)]:
            seq = sorted(items, key=lambda b: bbox(b)[axis])
            end = bbox(seq[0])[axis + 2]
            for i in range(1, len(seq)):
                gap = bbox(seq[i])[axis] - end
                if gap > threshold:
                    cuts.append((gap / threshold, seq[:i], seq[i:]))
                end = max(end, bbox(seq[i])[axis + 2])
        if cuts:
            _, a, b = max(cuts, key=lambda c: c[0])
            return walk(a) + walk(b)
        return sorted(items, key=lambda b: (bbox(b)[1], bbox(b)[0]))

    return walk(blocks)


def pdf_pages(source, out, ocr, selected):
    setup = Timings()
    bbox = out / "native-text.xhtml"
    setup.run(
        "native_extraction_s",
        lambda: subprocess.run(
            ["pdftotext", "-bbox-layout", str(source), str(bbox)], check=True
        ),
    )
    xml = setup.run("native_parse_s", lambda: ET.parse(bbox))
    pages = xml.findall(".//x:page", NS)
    pdf = setup.run("source_open_s", lambda: pdfium.PdfDocument(str(source)))
    try:
        if len(pages) != len(pdf):
            raise ValueError("PDF native/rendered page count mismatch")
        yield {"setup": setup.values, "count": len(pdf)}
        for n, element in enumerate(pages, 1):
            if selected and n not in selected:
                continue
            timing = Timings()
            uncertain = []

            def native():
                blocks = []
                lines = []
                for block in element.findall(".//x:block", NS):
                    current = []
                    for line in block.findall("x:line", NS):
                        text = clean(
                            " ".join(
                                "".join(w.itertext())
                                for w in line.findall("x:word", NS)
                            )
                        )
                        b = [
                            float(line.attrib[k])
                            for k in ["xMin", "yMin", "xMax", "yMax"]
                        ]
                        entry = {
                            "text": text,
                            "bbox_points": b,
                            "score": None,
                            "source": "native_pdf",
                        }
                        current.append(entry)
                        lines.append(entry)
                    if current:
                        blocks.append(current)
                heights = [
                    min(
                        l["bbox_points"][3] - l["bbox_points"][1],
                        l["bbox_points"][2] - l["bbox_points"][0],
                    )
                    for l in lines
                    if l["text"]
                ]
                median = statistics.median(heights) if heights else 10
                blocks = native_block_order(blocks)
                chunks = []
                for b in blocks:
                    text = "  \n".join(esc(l["text"]) for l in b)
                    if (
                        len(b) == 1
                        and len(b[0]["text"]) < 100
                        and b[0]["bbox_points"][3] - b[0]["bbox_points"][1]
                        > median * 1.6
                        and b[0]["bbox_points"][2] - b[0]["bbox_points"][0]
                        > b[0]["bbox_points"][3] - b[0]["bbox_points"][1]
                    ):
                        text = "### " + text
                    chunks.append(text)
                return blocks, lines, "\n\n".join(chunks)

            blocks, lines, text = timing.run("native_layout_s", native)
            page = pdf[n - 1]
            w, h = page.get_size()
            scale = 2000 / max(w, h)

            def render():
                bm = page.render(scale=scale)
                try:
                    return bm.to_numpy().copy()
                finally:
                    bm.close()

            active = []
            tp = None
            try:
                full = timing.run("page_render_s", render)
                full_path = f"pages/page-{n:04d}.jpg"
                timing.run(
                    "page_image_write_s",
                    lambda image=full: write_image(out / full_path, image),
                )
                del full

                def suppress():
                    nonlocal tp
                    tp = page.get_textpage()
                    kept = []
                    native_all = "".join(norm(line["text"]) for line in lines)
                    for obj in page.get_objects(
                        filter=[raw.FPDF_PAGEOBJ_TEXT], textpage=tp
                    ):
                        value = obj.extract()
                        if (
                            value.strip()
                            and "\ufffd" not in value
                            and "\x00" not in value
                            and norm(value)
                            and norm(value) in native_all
                        ):
                            if not raw.FPDFPageObj_SetIsActive(obj, False):
                                raise RuntimeError("Could not suppress PDF text")
                            active.append(obj)
                        elif value.strip():
                            kept.append(value)
                    return kept

                left = timing.run("native_suppression_s", suppress)
                artwork = timing.run("illustration_render_s", render)
            finally:
                for obj in active:
                    raw.FPDFPageObj_SetIsActive(obj, True)
                if tp is not None:
                    tp.close()
                page.close()
            art_path = f"illustrations/page-{n:04d}.jpg"
            timing.run(
                "illustration_write_s", lambda: write_image(out / art_path, artwork)
            )
            recognized, hit = ocr.recognize(artwork, timing)
            for l in lines:
                x0, y0, x1, y1 = l["bbox_points"]
                l["polygon"] = [
                    [x0 * scale, y0 * scale],
                    [x1 * scale, y0 * scale],
                    [x1 * scale, y1 * scale],
                    [x0 * scale, y1 * scale],
                ]
            supplement, selection = timing.run(
                "ocr_layout_s",
                lambda: image_text_sections(
                    recognized, lines, f"PDF page {n}", uncertain
                ),
            )
            md = [
                f"## PDF page {n}",
                "",
                f"[View full page]({full_path}) · [Illustrations]({art_path})",
                "",
                text or "*No native text on this page.*",
            ]
            if supplement:
                md += ["", "### Text embedded in illustrations (OCR)", "", supplement]
            if uncertain:
                md += [
                    "",
                    f"> {len(uncertain)} uncertain illustration-text detections are retained in [uncertain OCR](uncertain-ocr.md).",
                ]
            record = {
                "physical_page": n,
                "source_document": source.name,
                "native_blocks": blocks,
                "native_geometry": "PDF points; top-left origin; x right; y down. polygon additionally maps into the OCR image pixels via render_scale.",
                "render_scale": scale,
                "render_size": [artwork.shape[1], artwork.shape[0]],
                "image": full_path,
                "illustration_image": art_path,
                "suppressed_text_objects": len(active),
                "unextractable_text_left_visible": left,
                "ocr_record": f'ocr/{recognized["key"]}.json',
                "ocr_selection": selection,
                "cache_hit": hit,
            }
            yield {
                "number": n,
                "markdown": "\n".join(md),
                "record": record,
                "uncertain": uncertain,
                "timing": timing.values,
                "native_chars": sum(len(l["text"]) for l in lines),
                "ocr_lines": len(selection["included"]),
                "cache_hit": hit,
            }
    finally:
        pdf.close()


def local_member(base, href):
    if not href:
        raise ValueError("Missing archive resource")
    u = urlsplit(href)
    if u.scheme or u.netloc:
        raise ValueError("External resource not fetched: " + href)
    member = posixpath.normpath(
        posixpath.join(posixpath.dirname(base), unquote(u.path))
    )
    if member in (".", "..") or member.startswith(("/", "../")) or "\\" in member:
        raise ValueError("Unsafe archive resource: " + href)
    return member


def parse_xml(z, name):
    if z.getinfo(name).file_size > 200_000_000:
        raise ValueError("XML expansion bound")
    data = z.read(name)
    if b"<!ENTITY" in data.replace(b"\x00", b"").upper():
        raise ValueError("XML entity declaration rejected")
    return ET.fromstring(data)


def epub_spine(z):
    if sum(i.file_size for i in z.infolist()) > 3_000_000_000 or any(
        i.file_size > 200_000_000 for i in z.infolist()
    ):
        raise ValueError("Archive expansion bound")
    container = parse_xml(z, "META-INF/container.xml")
    package = local_member(
        "",
        next(
            n.attrib["full-path"]
            for n in container.iter()
            if n.tag.endswith("rootfile")
        ),
    )
    opf = parse_xml(z, package)
    manifest = {n.attrib["id"]: n.attrib for n in opf.iter() if n.tag.endswith("}item")}
    refs = [n.attrib for n in opf.iter() if n.tag.endswith("}itemref")]
    docs = [
        {
            "href": local_member(package, manifest[r["idref"]]["href"]),
            "linear": r.get("linear", "yes"),
        }
        for r in refs
    ]
    return package, docs


def epub_lines(tree):
    lines = []
    containers = [
        e for e in tree.iter() if e.attrib.get("id", "").startswith("TextContainer")
    ]

    def append(e, parent):
        text = clean("".join(e.itertext()))
        if not text:
            return
        style = e.attrib.get("style", "")
        left = re.search(r"(?:^|;)\s*left:([^;]+)", style)
        parent_left = re.search(
            r"(?:^|;)\s*left:([^;]+)", parent.attrib.get("style", "")
        )
        classes = e.attrib.get("class", "")
        size = next((c for c in classes.split() if c.startswith("fs")), None)
        lines.append(
            {
                "text": text,
                "source": "native_xhtml",
                "score": None,
                "polygon": None,
                "element_id": e.attrib.get("id"),
                "tag": e.tag.split("}")[-1],
                "style": style,
                "parent_style": parent.attrib.get("style", ""),
                "classes": classes,
                "group": (
                    (
                        left.group(1)
                        if left
                        else parent_left.group(1) if parent_left else None
                    ),
                    size,
                ),
                "links": [
                    a.attrib["href"]
                    for a in e.iter()
                    if a.tag.split("}")[-1] == "a" and "href" in a.attrib
                ],
            }
        )

    def visit(e, parent):
        tag = e.tag.split("}")[-1]
        if tag in ("script", "style", "img", "image"):
            return
        if tag in ("span", "p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th"):
            append(e, parent)
            return
        children = list(e)
        if tag in ("div", "body") and not any(
            c.tag.split("}")[-1]
            in (
                "div",
                "section",
                "article",
                "p",
                "h1",
                "h2",
                "h3",
                "h4",
                "h5",
                "h6",
                "ul",
                "ol",
                "table",
            )
            for c in children
        ):
            append(e, parent)
            return
        if e.text and e.text.strip():
            lines.append(
                {
                    "text": clean(e.text),
                    "source": "native_xhtml",
                    "score": None,
                    "polygon": None,
                    "element_id": e.attrib.get("id"),
                    "tag": tag,
                    "group": None,
                    "links": [],
                }
            )
        for child in e:
            visit(child, e)
            if child.tail and child.tail.strip():
                lines.append(
                    {
                        "text": clean(child.tail),
                        "source": "native_xhtml",
                        "score": None,
                        "polygon": None,
                        "element_id": None,
                        "tag": tag,
                        "group": None,
                        "links": [],
                    }
                )

    if containers:
        for c in containers:
            for child in c:
                visit(child, c)
    else:
        body = next((e for e in tree.iter() if e.tag.split("}")[-1] == "body"), tree)
        visit(body, body)
    chunks = []
    last = None
    for line in lines:
        text = esc(line["text"])
        if line["tag"] in ("h1", "h2", "h3", "h4", "h5", "h6"):
            text = "### " + text
        elif line["tag"] == "li":
            text = "- " + text
        if last != line.get("group") or not chunks:
            chunks.append(text)
        else:
            chunks[-1] += "  \n" + text
        last = line.get("group")
    return lines, "\n\n".join(chunks)


def epub_pages(source, out, ocr, selected):
    setup = Timings()
    z = setup.run("source_open_s", lambda: zipfile.ZipFile(source))
    try:
        package, docs = setup.run("native_extraction_s", lambda: epub_spine(z))
        (out / "spine.json").write_text(
            json.dumps({"package": package, "spine": docs}, indent=2)
        )
        yield {"setup": setup.values, "count": len(docs)}
        for n, doc in enumerate(docs, 1):
            if selected and n not in selected:
                continue
            timing = Timings()
            uncertain = []
            href = doc["href"]
            tree = timing.run("native_parse_s", lambda: parse_xml(z, href))
            lines, text = timing.run("native_layout_s", lambda: epub_lines(tree))
            # Keep the original fixed-layout XHTML as non-executed provenance, not an external-resource preview.
            timing.run(
                "native_record_write_s",
                lambda: (out / "native" / f"section-{n:04d}.xhtml").write_bytes(
                    z.read(href)
                ),
            )
            md = [
                f"## EPUB section {n}",
                "",
                f"Source: {esc(href)}"
                + (" (non-linear resource)" if doc["linear"] == "no" else ""),
                "",
                text or "*No native text in this section.*",
            ]
            images = []
            sum_ocr = 0
            hits = 0
            for j, e in enumerate(
                [e for e in tree.iter() if e.tag.split("}")[-1] in ("img", "image")], 1
            ):
                resource = e.attrib.get("src") or e.attrib.get(
                    "{http://www.w3.org/1999/xlink}href"
                )
                member = local_member(href, resource)
                data = timing.run("image_read_s", lambda: z.read(member))

                def decode():
                    with Image.open(io.BytesIO(data)) as im:
                        if im.width * im.height > 40_000_000:
                            raise ValueError("Image pixel bound")
                        rgba = im.convert("RGBA")
                        base = Image.new("RGBA", rgba.size, "white")
                        base.alpha_composite(rgba)
                        return cv2.cvtColor(
                            np.asarray(base.convert("RGB")), cv2.COLOR_RGB2BGR
                        )

                image = timing.run("image_decode_s", decode)
                dest = f"illustrations/section-{n:04d}-{j:02d}{Path(member).suffix.lower()}"
                timing.run(
                    "illustration_write_s", lambda: (out / dest).write_bytes(data)
                )
                recognized, hit = ocr.recognize(image, timing)
                hits += hit
                supplement, selection = timing.run(
                    "ocr_layout_s",
                    lambda: image_text_sections(
                        recognized,
                        lines,
                        f"EPUB section {n} image {j}",
                        uncertain,
                        region="epub_section",
                    ),
                )
                sum_ocr += len(selection["included"])
                md += ["", f"[Illustration {j}]({dest})"]
                if supplement:
                    md += [
                        "",
                        f"### Text embedded in illustration {j} (OCR)",
                        "",
                        supplement,
                    ]
                images.append(
                    {
                        "source_href": member,
                        "path": dest,
                        "width": image.shape[1],
                        "height": image.shape[0],
                        "ocr_record": f'ocr/{recognized["key"]}.json',
                        "ocr_selection": selection,
                        "cache_hit": hit,
                        "element_attributes": e.attrib,
                    }
                )
            if uncertain:
                md += [
                    "",
                    f"> {len(uncertain)} uncertain illustration-text detections are retained in [uncertain OCR](uncertain-ocr.md).",
                ]
            ids = [e.attrib["id"] for e in tree.iter() if "id" in e.attrib]
            anchors = "\n".join(
                f'<a id="source-{n}-{hashlib.sha256(i.encode()).hexdigest()[:12]}"></a>'
                for i in ids
            )
            source_links = [
                e.attrib["href"]
                for e in tree.iter()
                if e.tag.split("}")[-1] == "a" and "href" in e.attrib
            ]
            if source_links:
                md += ["", "### Source link references", ""] + [
                    "- " + esc(l) for l in source_links
                ]
            record = {
                "spine_index": n,
                "spine_href": href,
                "linear": doc["linear"],
                "native_lines": lines,
                "native_geometry": "Not inferred. Original fixed-layout styles retained verbatim; polygon and score null. Native text follows source DOM grouping/spine order.",
                "source_ids": {
                    i: f"source-{n}-{hashlib.sha256(i.encode()).hexdigest()[:12]}"
                    for i in ids
                },
                "source_links": source_links,
                "images": images,
            }
            yield {
                "number": n,
                "markdown": anchors + "\n\n" + "\n".join(md),
                "record": record,
                "uncertain": uncertain,
                "timing": timing.values,
                "native_chars": sum(len(l["text"]) for l in lines),
                "ocr_lines": sum_ocr,
                "cache_hit": hits,
            }
    finally:
        z.close()
