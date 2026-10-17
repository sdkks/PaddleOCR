# uv environment for local OCR

For local operators and agents using this checkout on native Apple Silicon macOS.
Use an isolated environment; do not install into system Python or use an
unexamined Rosetta interpreter.

## Choose Python for the whole stack

Use **Python 3.12 on macOS 14 or newer** for the baseline below. Successful local
converter runs used Python 3.12.7, PaddleX 3.7.2, PaddlePaddle 3.3.1, ONNX Runtime
1.31.0 and pypdfium2 5.14.0 on native arm64. The ONNX Runtime 1.31.0 macOS arm64
wheel requires macOS 14+; this recipe is not suitable for older macOS releases.

Python 3.14 is not the recommendation for this whole tested stack:

- Published PaddlePaddle 3.3.1 wheels cover CPython 3.9–3.13, not 3.14; a uv
  resolution targeting macOS arm64/Python 3.14 found no compatible wheel.
- PaddleX 3.7.2 pins PyYAML 6.0.2; a binary-only Python 3.14 resolution also failed.
- Source builds and ONNX-only Python 3.14 configurations were not tested. These
  findings do not prove that every possible Python 3.14 OCR setup is impossible.

The root manifest's `requires-python = ">=3.8"` is not proof that every dependency
supports every newer interpreter. The checkout-local converter itself needs
Python 3.11+; choose a version compatible with both the tool and its dependencies.

## Create and install, with permission

Prerequisites: `uv` available, native arm64 execution, and this repository checked
out. Run from the repository root. Before creating anything, inspect any existing
`.venv`; reuse it only after verifying compatibility, or choose a different new
path and adjust every command. Do not overwrite or clear an existing environment.
The repository already ignores `.venv`.

**Agents must obtain permission before running these setup commands.** uv may
need to download Python, seed packages and dependencies. Package-download approval
is not permission to upload documents or download models.

For a new `.venv`:

```sh
uv venv .venv --python 3.12 --seed
uv pip install --python .venv/bin/python -e . \
  'paddlex[ocr-core]==3.7.2' \
  'paddlepaddle==3.3.1' \
  'onnxruntime==1.31.0' \
  'pypdfium2==5.14.0'
uv pip check --python .venv/bin/python
.venv/bin/python -B -m tools.local_book_conversion --help
```

`--seed` is optional: it installs one or more of pip, setuptools and wheel,
according to the selected Python/uv behavior. `uv pip` does not require pip inside
the environment. Activation is also optional; explicit interpreter paths avoid
accidentally running another Python. In bash or zsh, `source .venv/bin/activate`
selects the environment for subsequent `python` commands.

These exact core versions are a baseline example, not a complete lockfile or
permanent compatibility guarantee. The checkout manifest plus these pins resolved
successfully without builds on a native Python 3.12 host. A fresh installation of
this recipe was not executed; resolution is not runtime proof. `uv pip check`
checks dependency consistency, and `--help` does not load the OCR stack: neither
proves real-model inference works. Validate a synthetic image before real documents.
Recheck package support before changing Python or dependency versions.

This setup does not require `uv init`, `uv sync`, a packaging migration or a new
lockfile. Do not force incompatible dependencies or patch upstream packages to
make a different interpreter work.

## Optional `.python-version`

Prefer the explicit `--python 3.12` above when only this tool needs that version.
If you intentionally want a checkout-wide default for uv Python selection, run
from the repository root:

```sh
uv python pin 3.12
```

This writes `.python-version`. uv uses it for Python selection unless an explicit
`--python` overrides it. It is a version preference, not a dependency lockfile;
`3.12` does not pin a patch release. It does not upgrade or recreate an existing
virtual environment. Inspect any existing pin first and agree on the broader
checkout default before changing it. Do not pin 3.14 for this baseline.

## Continue with local conversion

Follow [CPU runtime guidance](cpu-runtime.md) for trusted matching ONNX model
directories and supported languages, then [local converter guidance](local-converter.md)
for invocation and output checks. PDF extraction also requires Poppler's
`pdftotext` on PATH; the Python setup above does not install it. Obtain native
tools and model artifacts separately with permission.

Use `.venv/bin/python` in place of `python` in converter examples if not activated.
Keep source documents, extracted text, images, profiles and logs outside the
checkout. Never place private document paths or corpus content in shared examples.

Official uv references: [Python environments](https://docs.astral.sh/uv/pip/environments/)
and [Python versions](https://docs.astral.sh/uv/concepts/python-versions/).
