"""Explicit local inputs, fresh output, sequential book processes."""

import argparse
import csv
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import resource
import re
import statistics
import subprocess
import sys
import time


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        required=True,
        type=Path,
        help="PDF/EPUB file or directory (nonrecursive)",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="New destination outside the checkout (and outside directory inputs)",
    )
    p.add_argument("--det-model-dir", required=True, type=Path)
    p.add_argument("--rec-model-dir", required=True, type=Path)
    p.add_argument("--det-model-name", default="PP-OCRv6_tiny_det")
    p.add_argument("--rec-model-name", default="PP-OCRv6_tiny_rec")
    return p


def validate(args):
    source = args.input.resolve(strict=True)
    output = args.output.absolute()
    repo = Path(__file__).resolve().parents[2]
    if output.exists() or output.is_symlink():
        raise ValueError("Output already exists; choose a fresh destination")
    output = output.resolve()
    if output.is_relative_to(repo):
        raise ValueError("Output must be outside the source checkout")
    if source.is_dir() and output.is_relative_to(source):
        raise ValueError("Output must be outside the input directory")
    sources = (
        sorted(
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in (".pdf", ".epub")
        )
        if source.is_dir()
        else [source]
    )
    if not sources or any(
        not p.is_file() or p.suffix.lower() not in (".pdf", ".epub") for p in sources
    ):
        raise ValueError("Input must contain PDF or EPUB files")
    for model in (args.det_model_dir, args.rec_model_dir):
        if not model.is_dir():
            raise ValueError("Both explicit local model directories must exist")
        if output == model.resolve() or model.resolve().is_relative_to(output):
            raise ValueError("Output must not contain model inputs")
    return sources, output


def configure_runtime(out):
    # Set before importing inference; direct supported caches outside the checkout.
    os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(out / "runtime-cache")
    os.environ["XDG_CACHE_HOME"] = str(out / "runtime-cache")
    sys.dont_write_bytecode = True


def model_config(args):
    return dict(
        text_detection_model_name=args.det_model_name,
        text_detection_model_dir=str(args.det_model_dir.resolve()),
        text_recognition_model_name=args.rec_model_name,
        text_recognition_model_dir=str(args.rec_model_dir.resolve()),
        device="cpu",
        engine="onnxruntime",
        engine_config={
            "providers": ["CPUExecutionProvider"],
            "intra_op_num_threads": 6,
            "inter_op_num_threads": 1,
            "execution_mode": "sequential",
        },
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_det_limit_type="max",
        text_det_limit_side_len=1536,
        text_det_thresh=0.3,
        text_det_box_thresh=0.6,
        text_det_unclip_ratio=1.5,
        text_recognition_batch_size=6,
        text_rec_score_thresh=0.0,
    )


def peak_rss():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def dependency_versions():
    versions = {"python": sys.version.split()[0]}
    for package in (
        "paddleocr",
        "paddlex",
        "onnxruntime",
        "pypdfium2",
        "numpy",
        "opencv-python",
        "opencv-contrib-python",
        "Pillow",
    ):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def validate_outputs(out, records, expected):
    if [record["number"] for record in records] != list(range(1, expected + 1)):
        raise ValueError("Incomplete or out-of-order unit coverage")
    paths = [record["record"] for record in records]
    for markdown in ("book.md", "uncertain-ocr.md"):
        paths.extend(
            re.findall(
                r"(?<!\\)\]\(([^)\n]+)\)", (out / markdown).read_text(encoding="utf-8")
            )
        )
    for index in records:
        record = json.loads((out / index["record"]).read_text(encoding="utf-8"))
        if "physical_page" in record:
            paths.extend(
                record[key] for key in ("image", "illustration_image", "ocr_record")
            )
        else:
            paths.append(f"native/section-{index['number']:04d}.xhtml")
            for image in record["images"]:
                paths.extend(image[key] for key in ("path", "ocr_record"))
    for path in paths:
        target = (out / path).resolve()
        if not target.is_relative_to(out.resolve()) or not target.is_file():
            raise ValueError("Missing or unsafe output link: " + path)
    return len(paths)


def convert(source, out, args, model_factory=None):
    start, cpu = time.perf_counter(), time.process_time()
    out.mkdir(parents=True, exist_ok=False)
    configure_runtime(out)
    imported = time.perf_counter()
    from .common import IllustrationOCR, Timings, esc
    from .formats import pdf_pages, epub_pages

    if model_factory is None:
        from paddleocr import PaddleOCR

        model_factory = PaddleOCR
    stages = Timings()
    stages.values["dependency_import_s"] = time.perf_counter() - imported
    for folder in ("pages", "illustrations", "records", "native"):
        (out / folder).mkdir()
    config = model_config(args)
    save_json(out / "ocr-config.json", config)

    def source_hash():
        with source.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    digest = stages.run("source_hash_s", source_hash)
    model = stages.run("model_initialization_s", lambda: model_factory(**config))
    rows, parts, uncertain, records = [], [], [], []
    generator = None
    try:
        ocr = IllustrationOCR(out, model)
        generator = (pdf_pages if source.suffix.lower() == ".pdf" else epub_pages)(
            source, out, ocr, None
        )
        with (out / "unit-timings.jsonl").open("w", encoding="utf-8") as log:
            while True:
                wall, unit_cpu = time.perf_counter(), time.process_time()
                try:
                    unit = next(generator)
                except StopIteration:
                    break
                if "setup" in unit:
                    stages.values.update(unit["setup"])
                    expected = unit["count"]
                    continue
                n = unit["number"]
                dest = f"records/unit-{n:04d}.json"
                stages.run(
                    "unit_record_write_s", lambda: save_json(out / dest, unit["record"])
                )
                parts.append(unit["markdown"])
                uncertain.extend(unit["uncertain"])
                records.append({"number": n, "record": dest})
                row = {k: unit[k] for k in ("native_chars", "ocr_lines", "cache_hit")}
                row.update(
                    unit=n,
                    **unit["timing"],
                    unit_total_s=time.perf_counter() - wall,
                    unit_cpu_s=time.process_time() - unit_cpu,
                    peak_rss_bytes=peak_rss(),
                )
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                log.flush()
    finally:
        try:
            if generator is not None:
                generator.close()
        finally:
            stages.run("model_close_s", model.close)

    def assemble():
        prefix = f"# {esc(source.stem)}\n\nNative text plus illustration OCR. Images preserve layout; OCR may be incorrect.\n\n[Uncertain OCR](uncertain-ocr.md)\n\n"
        (out / "book.md").write_text(
            prefix + "\n\n---\n\n".join(parts) + "\n", encoding="utf-8"
        )
        (out / "uncertain-ocr.md").write_text(
            "# Uncertain illustration OCR\n\nHypotheses below 0.65; scores are not accuracy. All rotations and alternatives remain in ocr/.\n\n"
            + "\n".join(uncertain)
            + "\n",
            encoding="utf-8",
        )
        save_json(out / "page-index.json", records)
        with (out / "unit-timings.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
            )
            writer.writeheader()
            writer.writerows(rows)

    stages.run("assembly_s", assemble)
    link_count = stages.run(
        "output_validation_s", lambda: validate_outputs(out, records, expected)
    )
    latencies = sorted(row["unit_total_s"] for row in rows)
    keys = {
        key
        for row in rows
        for key in row
        if key.endswith("_s") and key not in ("unit_total_s", "unit_cpu_s")
    }
    profile = dict(
        source=str(source),
        source_sha256=digest,
        format=source.suffix[1:].lower(),
        units=len(rows),
        stages=stages.values,
        unit_stage_totals={k: sum(r.get(k, 0) for r in rows) for k in sorted(keys)},
        latency=(
            {
                "mean_s": statistics.mean(latencies),
                "p50_s": statistics.median(latencies),
                "p95_s": latencies[math.ceil(0.95 * len(latencies)) - 1],
                "max_s": max(latencies),
            }
            if latencies
            else {}
        ),
        ocr_passes=ocr.calls,
        cache_hits=ocr.hits,
        unique_ocr_images=len(ocr.cache),
        peak_rss_bytes=peak_rss(),
        cpu_s=time.process_time() - cpu,
        script_wall_s=time.perf_counter() - start,
        dependencies=dependency_versions(),
        config=config,
        validation={
            "checked_output_links": link_count,
            "units": len(rows),
            "expected": expected,
            "native_characters": sum(r["native_chars"] for r in rows),
            "image_ocr_lines_in_markdown": sum(r["ocr_lines"] for r in rows),
            "uncertain_lines": len(uncertain),
        },
    )
    save_json(out / "profile.json", profile)
    return profile


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        sources, out = validate(args)
    except (ValueError, OSError) as error:
        p.error(str(error))
    if args.input.is_file():
        convert(sources[0], out, args)
        return
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    for number, source in enumerate(sources, 1):
        dest = out / f"book-{number:04d}"
        command = [
            sys.executable,
            "-B",
            "-m",
            "tools.local_book_conversion",
            "--input",
            str(source.resolve()),
            "--output",
            str(dest),
            "--det-model-dir",
            str(args.det_model_dir.resolve()),
            "--rec-model-dir",
            str(args.rec_model_dir.resolve()),
            "--det-model-name",
            args.det_model_name,
            "--rec-model-name",
            args.rec_model_name,
        ]
        start = time.perf_counter()
        with (out / f"book-{number:04d}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        rows.append(
            {
                "source": str(source),
                "output": dest.name,
                "exit_code": result.returncode,
                "process_wall_s": time.perf_counter() - start,
            }
        )
        save_json(out / "batch-profile.json", rows)
        if result.returncode:
            raise SystemExit(f"Conversion failed; see {out}/book-{number:04d}.log")


if __name__ == "__main__":
    main()
