# Optional checkout-local PDF / EPUB converter

Use this route only when the current repository contains
`tools/local_book_conversion/`. The skill does not install that tool, and other
extraction routes do not depend on it. Read the
[tool README](../../../../tools/local_book_conversion/README.md) for the complete
CLI, output contract and limitations; verify that it exists before proceeding.

## Select the route

The tool combines native PDF/EPUB text with OCR of illustrations and local timing
records. It does not rasterize native ebook text just to recognize it. Check the
native text first: the tool retains it and does not automatically judge its
accuracy. For scans with a defective prior OCR layer, follow the skill's scanned
PDF guidance rather than assuming this native-first converter repairs that layer.

EPUB extraction follows the original package spine, including non-linear entries.
Source XHTML, IDs and links are retained as provenance, not a complete browser
preview or a reconstructed navigable ebook. Layout, equations, tables and diagram
relationships still require visual checking against the linked images.

## Run from the repository root

Use an existing compatible Python **3.11+** environment. This Unix-oriented runner
uses `resource`; Apple Silicon macOS is the intended platform, not Windows.
Required dependencies are PaddleOCR/PaddleX, **ONNX Runtime CPU**, NumPy, OpenCV,
Pillow and pypdfium2. PDF extraction additionally needs Poppler `pdftotext` on PATH.
Do not install dependencies or download models automatically. Consult
[CPU runtime guidance](cpu-runtime.md) for matching local ONNX directories,
including their configuration/dictionary assets and supported source languages.

If you need a new environment, follow [uv environment setup](uv-environment.md)
for the Python 3.12 baseline on macOS 14+. Without activation, use
`.venv/bin/python` instead of `python` below.

```sh
python -B -m tools.local_book_conversion --help
python -B -m tools.local_book_conversion \
  --input ../documents/example.epub \
  --output ../converted/example-new \
  --det-model-dir ../models/detector-onnx \
  --rec-model-dir ../models/recognizer-onnx
```

All four path flags are required. Input is one PDF/EPUB or a nonrecursive directory
of them. Output must be a new destination outside the checkout and, for directory
input, outside the input directory. Optional `--det-model-name` and
`--rec-model-name` default to `PP-OCRv6_tiny_det` and `PP-OCRv6_tiny_rec`; names must
match the supplied ONNX artifacts. These models are not universally multilingual.
Batches use sequential fresh processes. There are no overwrite, resume, GPU,
page-selection or general tuning flags.

## Read and validate the results

Start with each book's `book.md` and linked images. Consult `uncertain-ocr.md`,
`records/`, `ocr/` and `page-index.json` for native/OCR source, geometry, scores,
rotation alternatives and deduplication decisions. Four right-angle OCR passes
are combined heuristically; repeated identical pixel images reuse results within
a book. Selected scores below 0.65 are kept in the uncertainty file rather than
ordinary Markdown. Higher scores are not proof of accuracy: artwork can look like
text, and tiny, curved or stylized lettering can remain partial or incorrect.

Coverage and generated local links are checked before a success profile is
written. These are structural checks, not semantic completeness or proofreading.
A missing success profile is not success. Failures leave partial outputs; choose
a new destination to retry. Batches stop at the first failed child.

Profiling is in `profile.json`, `unit-timings.jsonl` and `unit-timings.csv`; batch
roots add `batch-profile.json` and per-book logs. A unit means a PDF physical page
or EPUB spine section, not an invented EPUB printed page. Consult the README for
stage boundaries, CPU/subprocess accounting, cached-image costs and peak-RSS
semantics. Do not equate unit latency with whole-process time or turn a single run
into a universal performance claim. No GPU or energy measurement is implied.

## Privacy and execution limits

Generated Markdown, images, XHTML, profiles, logs and caches can contain document
content, source filenames and user-supplied source/model paths. Keep them outside
the repository and inspect/redact before sharing. Never copy real document data
into examples or tests. Local models and offline/cache hints express offline
intent, **not verified network isolation**; use a separately verified restricted
environment when strict isolation is required.

Do not execute retained XHTML/scripts or fetch embedded remote resources.
External image/resource URLs, escaping archive references and XML entity
declarations are rejected, but parsing bounds are not a hostile-document sandbox.
Synthetic tests cover native extraction and fake-OCR plumbing, not real-model
accuracy, network isolation or perfect conversion. No extra input formats, SVG
rasterizer or DRM bypass are provided by this tool.
