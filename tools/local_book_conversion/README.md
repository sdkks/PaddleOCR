# Local PDF / EPUB conversion

Optional checkout-local utility for native text plus illustration OCR. It is not
an installed command, part of the public PaddleOCR API, or a requirement for other
OCR/extraction routes. Run from this repository root with a compatible existing
Python environment. Sources are never rewritten.

## Requirements

- Python **3.11+** (`hashlib.file_digest` is used), on macOS or a compatible Unix
  system (`resource` supplies process measurements). Apple Silicon CPU is the
  intended environment; Windows is not supported by this runner.
- Compatible PaddleOCR and PaddleX with the `engine="onnxruntime"` pipeline API,
  **ONNX Runtime CPU**, NumPy, OpenCV, Pillow, and pypdfium2. These imports are
  currently shared by both input routes. This tool does not install dependencies.
- Poppler's `pdftotext` on `PATH` for PDF native extraction (`-bbox-layout`).
- Explicit local **matching ONNX detection and recognition model directories**,
  including configuration/dictionary assets. Raw `.onnx` files or Paddle-format
  models are not interchangeable. Choose names matching those artifacts and the
  source language. The default tiny recognizer is not universally multilingual.

Consult the existing [CPU runtime guidance](../../.agents/skills/darwin-arm64-ocr/references/cpu-runtime.md)
for model selection and runtime compatibility. No dependency/lockfile changes or
model downloads are performed by this utility. Install and obtain trusted models
separately with permission before conversion.

For a new environment, see [uv environment setup](../../.agents/skills/darwin-arm64-ocr/references/uv-environment.md)
for the Python 3.12 baseline on macOS 14+, Python 3.14 limitations and optional
`.python-version` behavior. Without activation, use `.venv/bin/python` instead of
`python` in the examples below.

## CLI

From the repository root, use your selected environment's `python`:

```sh
python -B -m tools.local_book_conversion --help
python -B -m tools.local_book_conversion \
  --input ../documents/example.pdf \
  --output ../converted/example-new \
  --det-model-dir ../models/detector-onnx \
  --rec-model-dir ../models/recognizer-onnx
```

All four path flags are required. `--input` accepts one `.pdf`/`.epub`, or a
directory: only immediate PDF/EPUB files are selected, sorted by path, without
recursion. Suffix matching is case-insensitive; suffixes do not establish actual
format validity. For an EPUB, change the example input to `example.epub`.

Optional model names:

| Flag | Default |
| --- | --- |
| `--det-model-name` | `PP-OCRv6_tiny_det` |
| `--rec-model-name` | `PP-OCRv6_tiny_rec` |

There are no page-selection, resume, overwrite, worker-pool, GPU or tuning flags.
The output destination must not exist (including dangling symlinks), must be
outside this checkout and, for directory input, outside that input directory.
A batch uses sequential fresh Python processes, one per book; output subdirectories
are `book-0001`, `book-0002`, etc. Inputs/models remain user-managed.

## Processing and output

PDF native text comes from `pdftotext`, not OCR of rasterized native text. The tool
renders a full-page JPEG for inspection, suppresses PDF text objects only when
matching extracted text, and renders remaining artwork for OCR. Unextractable
text stays visible. The longest render side is 2000 pixels. Native blocks use a
whitespace-based ordering heuristic; check columns and rotated text visually.

EPUB follows package spine order, including non-linear entries. XHTML headings,
paragraphs and basic grouping become readable Markdown; native text is **not
rasterized for OCR**. Referenced `img` / SVG `image` resources are decoded and OCR'd
as illustrations. Original XHTML, source IDs and link references remain as
provenance, not as a browser preview or a complete EPUB renderer. External resource
URLs and escaping member paths are rejected, not fetched. XML entity declarations
and oversized archives/images are rejected. These checks are bounds, not a hostile
document sandbox. Do not execute retained XHTML or follow its links blindly.

The fixed inference settings use ONNX Runtime CPU with six intra-op threads, one
inter-op thread and sequential execution; recognition batch size six and detection
max side 1536. Document orientation classification, unwarping and text-line
orientation models are disabled. Each unique decoded illustration is submitted at
0°, 90°, 180° and 270°; coordinates map back to the original image. Identical pixel
images reuse cached results within a book.

Each book directory contains:

- `book.md`: native text, linked images and selected illustration OCR.
- `uncertain-ocr.md`: selected low-score hypotheses (below 0.65).
- `pages/`: full PDF page JPEGs; empty for EPUB.
- `illustrations/`: PDF native-suppressed JPEGs or copied EPUB image resources.
- `ocr/`: raw hypotheses from every rotation plus selected alternatives, polygons,
  scores and per-pass wall timings, keyed by image hash.
- `records/unit-NNNN.json` and `page-index.json`: PDF physical page or EPUB spine
  unit provenance, native text geometry when available, source links and OCR links.
- PDF `native-text.xhtml`, or EPUB `spine.json` and `native/section-NNNN.xhtml`.
- `ocr-config.json`, `profile.json`, `unit-timings.jsonl`, `unit-timings.csv`.
- `runtime-cache/` if the underlying runtime creates cache files.

Batch roots also contain `batch-profile.json` and per-book combined stdout/stderr
logs (`book-NNNN.log`). Coverage and generated local output links are checked before
a successful book profile is written. These checks prove structural coverage and
file existence, **not** correctness or completeness of recognized text.

## Profiling and privacy

`profile.json` records the supplied source path and SHA-256, selected configuration,
Python/distribution versions (null if distribution metadata is unavailable), unit
counts, cache/pass counts, wall/CPU stages, per-unit stage totals, latency mean,
median, nearest-rank p95 and maximum, and process high-water RSS in bytes.
`unit-timings.jsonl` is flushed per completed unit; CSV is assembled on success.
A unit is a physical PDF page or EPUB spine section, not an EPUB printed page.

Unit totals time extraction/render/recognition and record writes, not model startup
or final assembly. Stage fields ending `_s` are wall seconds; `_s_cpu` fields are
process CPU seconds. CPU time does not include child `pdftotext` work. Stage totals
are not independent end-to-end benchmarks; cached repeated images have different
costs. `script_wall_s` excludes interpreter startup and final profile serialization;
batch `process_wall_s` includes the whole child process. Peak RSS is a process
high-water mark, not a sum or per-stage allocation measurement. No energy/GPU
measurement or universal speed claim is made.

**Generated outputs are private local data.** They include document text, images,
source filenames/paths and local model paths; logs may contain runtime diagnostics.
Do not copy profiles, caches or converted books into the repository or publish them
without inspecting/redacting them. The tool sets offline/cache hints and supplies
local models. This expresses offline intent; it is **not proven network isolation**
and does not sandbox third-party libraries. For strict isolation, use a separately
verified network-restricted environment.

## Limits and failures

- No perfect OCR guarantee: missed regions, incorrect characters and incorrect
  reading order remain possible. Scores are not calibrated accuracy estimates.
- Overlapping rotation results and native/OCR text are deduplicated heuristically.
  Duplicated rotations, alternative interpretations and over-/under-deduplication
  can remain; inspect raw records and source images when meaning matters.
- Native PDF text can already be erroneous hidden OCR. EPUB layout/CSS, tables,
  equations, diagram meaning and all navigation semantics are not reconstructed.
  Native EPUB geometry/confidence remain null instead of being invented. Source
  link references are retained, not fully rewritten into a navigable ebook.
- Only PDF and EPUB are supported. No DRM bypass, MOBI/AZW3/DjVu decoding, remote
  resources, SVG rasterizer or conversion plugins are provided.
- Failure leaves **partial outputs** for diagnosis. No rollback, overwrite or resume
  is implemented; choose a new destination to retry. A batch stops at its first
  failing child and records that exit code. Missing `profile.json` is not success.
- Model/runtime compatibility must be checked in the selected environment. Synthetic
  tests exercise native PDF/EPUB extraction and fake-OCR plumbing, not real model
  accuracy, real-model end-to-end compatibility or network isolation.

## Focused checks

With the native dependencies and `pdftotext` already available:

```sh
python -B -m unittest discover -s tools/local_book_conversion/tests -v
python -B -m tools.local_book_conversion --help
```

The tests create only synthetic documents/images in temporary directories, fake all
model inference, and do not require model downloads. The help path imports no OCR,
NumPy, OpenCV, Pillow or PDFium modules. Follow the repository pre-commit style
configuration for changed Python files; no training/inference suite is needed to
run these focused tests.
