"""Shared local-only extraction and illustration OCR helpers."""

import hashlib, json, re, time, unicodedata, statistics
from difflib import SequenceMatcher
import numpy as np


def clean(text):
    return re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", text).replace("\xad", "")
    ).strip()


def norm(text):
    return "".join(c for c in clean(text).casefold() if c.isalnum())


def esc(text):
    value = re.sub(r"([\\`*_{}\[\]<>#|])", r"\\\1", clean(text))
    return re.sub(r"^(\d+)\.(?=\s|$)", r"\1\\.", value)


def bounds(poly):
    return [
        min(p[0] for p in poly),
        min(p[1] for p in poly),
        max(p[0] for p in poly),
        max(p[1] for p in poly),
    ]


def overlap(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    bb = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (aa + bb - inter), inter / min(aa, bb), min(aa, bb) / max(aa, bb)


def order(lines):
    if len(lines) < 2:
        return lines
    h = statistics.median(
        max(1, bounds(l["polygon"])[3] - bounds(l["polygon"])[1]) for l in lines
    )
    cuts = []
    for axis, threshold in [(1, h * 0.9), (0, h)]:
        seq = sorted(lines, key=lambda l: bounds(l["polygon"])[axis])
        end = bounds(seq[0]["polygon"])[axis + 2]
        for i in range(1, len(seq)):
            gap = bounds(seq[i]["polygon"])[axis] - end
            if gap > threshold:
                cuts.append((gap / threshold, seq[:i], seq[i:]))
            end = max(end, bounds(seq[i]["polygon"])[axis + 2])
    if cuts:
        _, a, b = max(cuts, key=lambda c: c[0])
        return order(a) + order(b)
    return sorted(
        lines,
        key=lambda l: (
            round(bounds(l["polygon"])[1] / max(h * 0.6, 1)),
            bounds(l["polygon"])[0],
        ),
    )


class Timings:
    def __init__(self):
        self.values = {}

    def run(self, key, fn):
        t = time.perf_counter()
        c = time.process_time()
        value = fn()
        self.values[key] = self.values.get(key, 0) + time.perf_counter() - t
        self.values[key + "_cpu"] = (
            self.values.get(key + "_cpu", 0) + time.process_time() - c
        )
        return value


class IllustrationOCR:
    def __init__(self, root, model):
        self.root = root
        self.model = model
        self.cache = {}
        self.calls = 0
        self.hits = 0
        (root / "ocr").mkdir(exist_ok=True)

    def recognize(self, image, timing):
        h, w = image.shape[:2]
        key = timing.run(
            "image_hash_s",
            lambda: hashlib.sha256(
                str(image.shape).encode() + image.tobytes()
            ).hexdigest(),
        )
        if key in self.cache:
            self.hits += 1
            return self.cache[key], True
        detections = []
        passes = []
        for k in range(4):
            rotated = np.ascontiguousarray(np.rot90(image, k))
            t = time.perf_counter()
            result = timing.run(
                "image_ocr_s", lambda: next(iter(self.model.predict_iter(rotated)))
            )
            passes.append(
                {"rotation_ccw_degrees": k * 90, "wall_s": time.perf_counter() - t}
            )
            self.calls += 1
            for text, score, poly in zip(
                result["rec_texts"],
                result["rec_scores"],
                result["rec_polys"],
                strict=True,
            ):
                submitted = poly.tolist()
                mapped = []
                for x, y in submitted:
                    mapped.append(
                        [x, y]
                        if k == 0
                        else (
                            [w - y, x]
                            if k == 1
                            else [w - x, h - y] if k == 2 else [y, h - x]
                        )
                    )
                detections.append(
                    {
                        "text": text,
                        "score": float(score),
                        "polygon": mapped,
                        "submitted_polygon": submitted,
                        "rotation_ccw_degrees": 90 * k,
                    }
                )
        # Keep every raw hypothesis; only choose one overlapping interpretation for readable output.
        groups = []
        for i, line in sorted(enumerate(detections), key=lambda t: -t[1]["score"]):
            b = bounds(line["polygon"])
            found = None
            for group in groups:
                other = detections[group[0]]
                iou, cover, ratio = overlap(b, bounds(other["polygon"]))
                if iou > 0.45 or (
                    cover > 0.75
                    and ratio > 0.35
                    and SequenceMatcher(
                        None, norm(line["text"]), norm(other["text"])
                    ).ratio()
                    > 0.6
                ):
                    found = group
                    break
            if found is None:
                groups.append([i])
            else:
                found.append(i)
        selected = [{**detections[g[0]], "raw_ids": g} for g in groups]
        result = {
            "key": key,
            "width": w,
            "height": h,
            "coordinate_system": "original submitted image pixels; top-left origin; x right; y down. Each raw submitted_polygon is in its rotated input; polygon is inverse-rotated to original image. No OCR score filtering.",
            "passes": passes,
            "raw": detections,
            "selected": order(selected),
        }
        timing.run(
            "ocr_record_write_s",
            lambda: (self.root / "ocr" / f"{key}.json").write_text(
                json.dumps(result, ensure_ascii=False)
            ),
        )
        self.cache[key] = result
        return result, False


def image_text_sections(result, native_lines, unit_id, uncertain, region=None):
    included = []
    duplicates = []
    for line in result["selected"]:
        nt = norm(line["text"])
        duplicate = False
        if nt:
            for native in native_lines:
                nn = norm(native["text"])
                match = (
                    nt == nn
                    or (len(nt) >= 8 and nt in nn)
                    or (len(nt) >= 8 and SequenceMatcher(None, nt, nn).ratio() > 0.92)
                )
                if not match:
                    continue
                if native.get("polygon"):
                    _, cover, _ = overlap(
                        bounds(line["polygon"]), bounds(native["polygon"])
                    )
                    if cover > 0.25:
                        duplicate = True
                        break
                # For fixed-layout EPUB, dedup only long exact lines on this same image-bearing section.
                elif region == "epub_section" and len(nt) >= 20 and nt == nn:
                    duplicate = True
                    break
        if duplicate:
            duplicates.append(line)
            continue
        if line["score"] < 0.65:
            uncertain.append(
                f'- {unit_id}: {esc(line["text"])} (score {line["score"]:.2f}, image `{result["key"][:12]}`)'
            )
        else:
            included.append(line)
    text = "\n\n".join(esc(l["text"]) for l in included)
    return text, {
        "included": included,
        "duplicate_native": duplicates,
        "uncertain_count": sum(l["score"] < 0.65 for l in result["selected"]),
    }
