"""Synthetic fixtures only; no real models, network, or document corpus."""

import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image

from tools.local_book_conversion import __main__ as cli
from tools.local_book_conversion.common import (
    IllustrationOCR,
    Timings,
    image_text_sections,
)
from tools.local_book_conversion.formats import (
    epub_lines,
    local_member,
    parse_xml,
    native_block_order,
)


class FakeOCR:
    def __init__(self, **config):
        self.config = config
        self.calls = 0
        self.closed = False

    def predict_iter(self, image):
        self.calls += 1
        h, w = image.shape[:2]
        # A whole-image box is the same region after all inverse rotations.
        yield {
            "rec_texts": ["Synthetic illustration label", "Uncertain synthetic label"],
            "rec_scores": [0.9, 0.4],
            "rec_polys": [
                np.array([[0, 0], [w, 0], [w, h], [0, h]]),
                np.array([[1, 1], [3, 1], [3, 3], [1, 3]]),
            ],
        }

    def close(self):
        self.closed = True


def make_pdf(path):
    """Write one small text/vector PDF using only standard PDF objects."""
    stream = b"BT /F1 16 Tf 20 160 Td (Synthetic native text) Tj ET\n20 20 80 50 re S\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode()
        + b" >>\nstream\n"
        + stream
        + b"endstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for n, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{n} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(data)


def make_epub(path, image_href="images/chart.png"):
    image = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(image, format="PNG")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(
            "META-INF/container.xml",
            '<container><rootfiles><rootfile full-path="OPS/package.opf"/></rootfiles></container>',
        )
        z.writestr(
            "OPS/package.opf",
            """<package xmlns="http://www.idpf.org/2007/opf"><manifest>
<item id="a" href="a.xhtml"/><item id="b" href="b.xhtml"/>
</manifest><spine><itemref idref="b"/><itemref idref="a" linear="no"/></spine></package>""",
        )
        for name, text in (
            ("a", "Synthetic second section"),
            ("b", "Synthetic first section"),
        ):
            z.writestr(
                f"OPS/{name}.xhtml",
                f"""<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1 id="title">{text}</h1><p>Native paragraph <em>inline</em> tail.</p>
<img src="{image_href}"/><a href="a.xhtml#title">Cross reference</a>
</body></html>""",
            )
        z.writestr("OPS/images/chart.png", image.getvalue())


class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.det = self.root / "detector"
        self.rec = self.root / "recognizer"
        self.det.mkdir()
        self.rec.mkdir()
        self.environment = mock.patch.dict(os.environ)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        old = sys.dont_write_bytecode
        self.addCleanup(setattr, sys, "dont_write_bytecode", old)

    def args(self, source, output=None):
        return cli.parser().parse_args(
            [
                "--input",
                str(source),
                "--output",
                str(output or self.root / "result"),
                "--det-model-dir",
                str(self.det),
                "--rec-model-dir",
                str(self.rec),
            ]
        )

    def run_conversion(self, suffix):
        source = self.root / ("synthetic" + suffix)
        (make_pdf if suffix == ".pdf" else make_epub)(source)
        original = source.read_bytes()
        args = self.args(source)
        sources, out = cli.validate(args)
        model = FakeOCR()
        profile = cli.convert(sources[0], out, args, lambda **kw: model)
        self.assertTrue(model.closed)
        self.assertEqual(original, source.read_bytes())
        self.assertEqual(profile["source_sha256"], hashlib.sha256(original).hexdigest())
        self.assertEqual(profile["config"]["engine"], "onnxruntime")
        self.assertEqual(
            profile["config"]["engine_config"]["providers"], ["CPUExecutionProvider"]
        )
        self.assertEqual(profile["dependencies"]["python"], sys.version.split()[0])
        self.assertGreater(profile["validation"]["checked_output_links"], 0)
        rows = [
            json.loads(line)
            for line in (out / "unit-timings.jsonl").read_text().splitlines()
        ]
        self.assertEqual(len(rows), profile["units"])
        with (out / "unit-timings.csv").open() as f:
            self.assertEqual(len(list(csv.DictReader(f))), len(rows))
        self.assertGreaterEqual(profile["script_wall_s"], profile["latency"]["max_s"])
        self.assertGreater(profile["peak_rss_bytes"], 0)
        self.assertIn("output_validation_s", profile["stages"])
        self.assertIn(
            "Uncertain synthetic label", (out / "uncertain-ocr.md").read_text()
        )
        self.assertNotIn("Uncertain synthetic label", (out / "book.md").read_text())
        self.assertIn("Synthetic illustration label", (out / "book.md").read_text())
        self.assertEqual(profile, json.loads((out / "profile.json").read_text()))
        return out, profile

    def test_pdf_native_illustration_and_profile(self):
        out, profile = self.run_conversion(".pdf")
        self.assertEqual(profile["units"], 1)
        self.assertEqual(profile["ocr_passes"], 4)
        self.assertIn("Synthetic native text", (out / "book.md").read_text())
        record = json.loads((out / "records/unit-0001.json").read_text())
        self.assertEqual(record["physical_page"], 1)
        self.assertEqual(record["suppressed_text_objects"], 1)
        self.assertEqual(record["unextractable_text_left_visible"], [])
        self.assertEqual(record["native_blocks"][0][0]["source"], "native_pdf")
        self.assertNotEqual(
            (out / record["image"]).read_bytes(),
            (out / record["illustration_image"]).read_bytes(),
        )

    def test_epub_spine_native_images_cache_and_profile(self):
        out, profile = self.run_conversion(".epub")
        text = (out / "book.md").read_text()
        self.assertLess(
            text.index("Synthetic first section"),
            text.index("Synthetic second section"),
        )
        self.assertIn("Native paragraph inline tail.", text)
        self.assertIn("non-linear resource", text)
        self.assertEqual(profile["units"], 2)
        self.assertEqual(profile["cache_hits"], 1)
        self.assertEqual(profile["ocr_passes"], 4)
        first = json.loads((out / "records/unit-0001.json").read_text())
        self.assertEqual(first["spine_href"], "OPS/b.xhtml")
        self.assertIn("title", first["source_ids"])
        self.assertEqual(first["source_links"], ["a.xhtml#title"])
        self.assertIsNone(first["native_lines"][0]["polygon"])

    def test_existing_output_is_never_overwritten(self):
        source = self.root / "synthetic.epub"
        make_epub(source)
        args = self.args(source)
        args.output.mkdir()
        marker = args.output / "marker"
        marker.write_text("keep")
        with self.assertRaisesRegex(ValueError, "already exists"):
            cli.validate(args)
        with self.assertRaises(FileExistsError):
            cli.convert(source, args.output, args, FakeOCR)
        self.assertEqual(marker.read_text(), "keep")

    def test_output_location_and_local_models(self):
        source = self.root / "synthetic.epub"
        make_epub(source)
        repo = Path(cli.__file__).resolve().parents[2]
        with self.assertRaisesRegex(ValueError, "checkout"):
            cli.validate(self.args(source, repo / "synthetic-not-created"))
        with self.assertRaisesRegex(ValueError, "input directory"):
            cli.validate(self.args(self.root, self.root / "nested"))
        args = self.args(source)
        args.det_model_dir = self.root / "missing"
        with self.assertRaisesRegex(ValueError, "directories"):
            cli.validate(args)

    def test_symlink_output_rejected(self):
        source = self.root / "synthetic.epub"
        make_epub(source)
        args = self.args(source)
        args.output.symlink_to(self.root / "nonexistent")
        with self.assertRaisesRegex(ValueError, "already exists"):
            cli.validate(args)

    def test_invalid_input_rejected(self):
        source = self.root / "synthetic.txt"
        source.write_text("synthetic")
        with self.assertRaisesRegex(ValueError, "PDF or EPUB"):
            cli.validate(self.args(source))

    def test_failure_closes_model_and_retains_partial_output(self):
        source = self.root / "synthetic.epub"
        make_epub(source, "https://example.invalid/remote.png")
        args = self.args(source)
        model = FakeOCR()
        with self.assertRaisesRegex(ValueError, "External resource"):
            cli.convert(source, args.output, args, lambda **kw: model)
        self.assertTrue(model.closed)
        self.assertTrue((args.output / "ocr-config.json").is_file())
        self.assertFalse((args.output / "profile.json").exists())
        with self.assertRaisesRegex(ValueError, "already exists"):
            cli.validate(args)

    def test_ocr_failure_closes_model(self):
        source = self.root / "synthetic.epub"
        make_epub(source)
        args = self.args(source)
        model = FakeOCR()
        model.predict_iter = mock.Mock(side_effect=RuntimeError("synthetic failure"))
        with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
            cli.convert(source, args.output, args, lambda **kw: model)
        self.assertTrue(model.closed)

    def test_link_and_coverage_validation(self):
        out, _ = self.run_conversion(".epub")
        records = json.loads((out / "page-index.json").read_text())
        with self.assertRaisesRegex(ValueError, "coverage"):
            cli.validate_outputs(out, list(reversed(records)), 2)
        (out / "book.md").write_text("[Missing](missing.jpg)")
        with self.assertRaisesRegex(ValueError, "output link"):
            cli.validate_outputs(out, records, 2)
        (out / "book.md").write_text("[Escape](../synthetic.epub)")
        with self.assertRaisesRegex(ValueError, "output link"):
            cli.validate_outputs(out, records, 2)

    def test_batch_sequential_process_contract(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        make_epub(inputs / "b.epub")
        make_epub(inputs / "a.epub")
        (inputs / "ignored.txt").write_text("synthetic")
        args = self.args(inputs)
        commands = []

        def child(command, **kw):
            commands.append(command)
            child_args = cli.parser().parse_args(command[4:])
            cli.convert(child_args.input, child_args.output, child_args, FakeOCR)
            return subprocess.CompletedProcess(command, 0)

        with mock.patch.object(cli.subprocess, "run", side_effect=child):
            cli.main(
                [
                    "--input",
                    str(inputs),
                    "--output",
                    str(args.output),
                    "--det-model-dir",
                    str(self.det),
                    "--rec-model-dir",
                    str(self.rec),
                ]
            )
        rows = json.loads((args.output / "batch-profile.json").read_text())
        self.assertEqual(
            [Path(row["source"]).name for row in rows], ["a.epub", "b.epub"]
        )
        self.assertEqual(len(commands), 2)
        self.assertEqual(
            commands[0][:4], [sys.executable, "-B", "-m", "tools.local_book_conversion"]
        )
        self.assertTrue(all(row["exit_code"] == 0 for row in rows))

    def test_batch_stops_and_records_failure(self):
        inputs = self.root / "inputs"
        inputs.mkdir()
        make_epub(inputs / "a.epub")
        make_epub(inputs / "b.epub")
        args = self.args(inputs)
        with mock.patch.object(
            cli.subprocess, "run", return_value=subprocess.CompletedProcess([], 7)
        ) as run:
            with self.assertRaisesRegex(SystemExit, "Conversion failed"):
                cli.main(
                    [
                        "--input",
                        str(inputs),
                        "--output",
                        str(args.output),
                        "--det-model-dir",
                        str(self.det),
                        "--rec-model-dir",
                        str(self.rec),
                    ]
                )
        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            json.loads((args.output / "batch-profile.json").read_text())[0][
                "exit_code"
            ],
            7,
        )

    def test_archive_entities_rejected(self):
        archive = self.root / "synthetic.zip"
        with zipfile.ZipFile(archive, "w") as z:
            z.writestr(
                "unsafe.xml",
                '<!DOCTYPE x [<!ENTITY x "expansion">]><x>&x;</x>'.encode("utf-16"),
            )
        with zipfile.ZipFile(archive) as z:
            with self.assertRaisesRegex(ValueError, "entity"):
                parse_xml(z, "unsafe.xml")


class HelperTests(unittest.TestCase):
    def test_epub_section_preserves_native_blocks(self):
        tree = ET.fromstring(
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><section><h1>Heading</h1><p>First paragraph.</p><p>Second paragraph.</p></section></body></html>'
        )
        lines, markdown = epub_lines(tree)
        self.assertEqual(
            [(line["tag"], line["text"]) for line in lines],
            [("h1", "Heading"), ("p", "First paragraph."), ("p", "Second paragraph.")],
        )
        self.assertEqual(
            markdown.splitlines(),
            ["### Heading  ", "First paragraph.  ", "Second paragraph."],
        )

    def test_epub_nested_article_preserves_native_blocks(self):
        tree = ET.fromstring(
            "<body><div><article><section><h2>Heading</h2>"
            "<p>First paragraph.</p></section><article><p>Second paragraph.</p>"
            "</article></article></div></body>"
        )
        lines, markdown = epub_lines(tree)
        self.assertEqual(
            [(line["tag"], line["text"]) for line in lines],
            [("h2", "Heading"), ("p", "First paragraph."), ("p", "Second paragraph.")],
        )
        self.assertTrue(markdown.startswith("### Heading  \n"))

    def test_epub_lower_headings_are_not_flattened(self):
        for wrapper in ("body", "div"):
            for tag in ("h4", "h5", "h6"):
                with self.subTest(wrapper=wrapper, tag=tag):
                    tree = ET.fromstring(
                        f"<html><{wrapper}><{tag}>Heading</{tag}>"
                        f"<{tag}>Next heading</{tag}></{wrapper}></html>"
                    )
                    lines, markdown = epub_lines(tree)
                    self.assertEqual(
                        [(line["tag"], line["text"]) for line in lines],
                        [(tag, "Heading"), (tag, "Next heading")],
                    )
                    self.assertEqual(
                        markdown.splitlines(), ["### Heading  ", "### Next heading"]
                    )

    def test_epub_inline_spans_join_without_inserted_spaces(self):
        for wrapper in ("body", "div", "p"):
            with self.subTest(wrapper=wrapper):
                tree = ET.fromstring(
                    f"<html><{wrapper}>pre<span>fix<span>ed</span></span> "
                    f"tail<span>s</span>.</{wrapper}></html>"
                )
                lines, markdown = epub_lines(tree)
                self.assertEqual([line["text"] for line in lines], ["prefixed tails."])
                self.assertEqual(markdown, "prefixed tails.")

    def test_epub_fixed_layout_text_container_still_groups_spans(self):
        tree = ET.fromstring(
            '<body><div id="TextContainer1" style="left:12px">'
            '<div><span class="fs12">pre<span>fix</span>ed tail.</span>'
            "</div><div><span>Next line.</span></div></div></body>"
        )
        lines, markdown = epub_lines(tree)
        self.assertEqual(
            [line["text"] for line in lines], ["prefixed tail.", "Next line."]
        )
        self.assertEqual([line["group"] for line in lines], [("12px", None)] * 2)
        self.assertEqual(markdown, "prefixed tail.  \nNext line.")

    def test_resource_paths(self):
        self.assertEqual(
            local_member("OPS/section.xhtml", "images/chart%20one.png#anchor"),
            "OPS/images/chart one.png",
        )
        for href in (
            "../../escape",
            "%2e%2e/%2e%2e/escape",
            "/absolute",
            "https://example.invalid/a",
            "//example.invalid/a",
            "bad\\path",
            None,
        ):
            with self.subTest(href=href), self.assertRaises(ValueError):
                local_member("OPS/section.xhtml", href)

    def test_rotations_cache_and_raw_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = FakeOCR()
            ocr = IllustrationOCR(root, model)
            image = np.zeros((20, 40, 3), dtype=np.uint8)
            result, hit = ocr.recognize(image, Timings())
            self.assertFalse(hit)
            self.assertEqual(len(result["raw"]), 8)
            self.assertEqual(
                [p["rotation_ccw_degrees"] for p in result["passes"]], [0, 90, 180, 270]
            )
            for line in result["raw"][::2]:
                self.assertEqual(
                    sorted(line["polygon"]), [[0, 0], [0, 20], [40, 0], [40, 20]]
                )
            again, hit = ocr.recognize(image.copy(), Timings())
            self.assertTrue(hit)
            self.assertEqual(again, result)
            self.assertEqual(model.calls, 4)
            saved = json.loads((root / "ocr" / (result["key"] + ".json")).read_text())
            self.assertEqual(saved, result)

    def test_native_duplicate_and_uncertain_selection(self):
        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        line = {"text": "Synthetic native long phrase", "polygon": poly, "score": 0.9}
        result = {"key": "synthetic", "selected": [line]}
        text, selection = image_text_sections(result, [line], "unit", [])
        self.assertEqual(text, "")
        self.assertEqual(selection["duplicate_native"], [line])
        native = {**line, "polygon": None}
        text, _ = image_text_sections(
            result, [native], "unit", [], region="epub_section"
        )
        self.assertEqual(text, "")
        uncertain = []
        text, _ = image_text_sections(
            {"key": "synthetic", "selected": [{**line, "score": 0.2}]},
            [],
            "unit",
            uncertain,
        )
        self.assertEqual(text, "")
        self.assertEqual(len(uncertain), 1)

    def test_native_columns_keep_blocks_together(self):
        def block(text, x, y):
            return [{"text": text, "bbox_points": [x, y, x + 30, y + 10]}]

        left = block("Left column", 0, 0)
        right = block("Right column", 100, 0)
        self.assertEqual(native_block_order([right, left]), [left, right])

    def test_help_is_lightweight(self):
        program = """import sys
from tools.local_book_conversion.__main__ import main
try:
    main(["--help"])
except SystemExit as error:
    assert error.code == 0
assert not {"paddleocr", "paddlex", "numpy", "cv2", "pypdfium2", "PIL"}.intersection(sys.modules)
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", program], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--det-model-dir", result.stdout)

    def test_cli_missing_arguments(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(
            SystemExit
        ) as error:
            cli.main([])
        self.assertEqual(error.exception.code, 2)

    def test_repository_sources_contain_no_machine_paths(self):
        root = Path(cli.__file__).parent
        forbidden = ("/" + "Users/", "/" + "home/", "/" + "tmp/")
        for file in root.rglob("*"):
            if file.suffix in (".py", ".md"):
                text = file.read_text()
                for value in forbidden:
                    self.assertNotIn(value, text, str(file))


if __name__ == "__main__":
    unittest.main()
