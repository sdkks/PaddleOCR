# CPU-first PaddleOCR runtime

## Establish a compatible local environment

Use an isolated native-arm64 Python environment, not a global install or an
unexamined Rosetta interpreter. Python 3.12 is a practical starting point. Check the
checkout's manifest, installed package versions and official API docs before setup;
do not force incompatible pins, downgrade blindly, or patch upstream dependencies.
The configuration below targets the PaddleOCR API backed by PaddleX 3.7. A checked
baseline used PaddleX 3.7.2, PaddlePaddle 3.3.1, ONNX Runtime 1.31 and pypdfium2 5.14;
these are compatibility evidence, not permanent requirements or a universal lockfile.

For a permission-gated uv setup on macOS 14+, see
[uv environment setup](uv-environment.md), including Python 3.14 limitations and
optional `.python-version` behavior.

Verify macOS arm64 wheels and maintained releases for the required OCR dependencies,
CPU `onnxruntime` and, for controlled PDF rendering/extraction, `pypdfium2`. Follow
installation instructions matching the checkout; do not copy CUDA/GPU installation
commands onto macOS. Confirm a synthetic image works before processing documents.

Download only trusted official model artifacts when network access is permitted.
For ONNX Runtime, supply the **matching ONNX model directories**, including their
configuration/dictionary assets, not raw `.onnx` files or Paddle-format directories.
Selecting a model name or `engine="onnxruntime"` is not automatic model export.
Official ONNX detection/recognition downloads are linked from the repository's
[cross-platform model table](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/inference_deployment/cross_platform/android_deployment.en.md).
That table supplies model artifacts; its Android runtime instructions are not macOS
setup instructions. Record artifact origin/version and checksums for reproducibility.

## Starting configuration, not a universal optimum

A measured M2 Pro comparison favored official v6 tiny models on ONNX Runtime CPU.
Start here for supported languages and upright pages, then measure on the target
machine. Smaller models are not universally more accurate, faster or lower-energy.
Check the recognition model's actual script coverage; v6 tiny is not a universal
multilingual model (in particular, it does not support Japanese). Explicit model
names/directories override `lang` selection; changing `lang` alone will not replace
an explicitly selected recognizer. Choose a compatible language-specific model when
needed and revalidate its backend/artifact format.

Replace the example relative directories with your extracted ONNX directories:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv6_tiny_det",
    text_detection_model_dir="./models/PP-OCRv6_tiny_det_onnx_infer",
    text_recognition_model_name="PP-OCRv6_tiny_rec",
    text_recognition_model_dir="./models/PP-OCRv6_tiny_rec_onnx_infer",
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
    text_det_limit_side_len=960,
    text_det_thresh=0.3,
    text_det_box_thresh=0.6,
    text_det_unclip_ratio=1.5,
    text_recognition_batch_size=6,
    text_rec_score_thresh=0.0,
)
```

Keep this instance alive across pages. For each decoded BGR image, consume
`ocr.predict_iter(image)` and persist results incrementally; call `ocr.close()` in
a `finally` block when finished. Arrays carry no document/page provenance, so attach
it yourself. Retain `rec_texts`, `rec_scores` and their aligned `rec_polys`; convert
NumPy values to ordinary JSON values if serializing. Do not align filtered text
against all `dt_polys`. Keeping recognition scores does not recover missed detections.

## PDFs without temporary PNGs

For controlled rendering, stream one page at a time through PDFium → BGR NumPy →
OCR. Start around 200 dpi (`scale=200/72` for ordinary PDF canvas units); visually
check tiny text and account for unusual page units/sizes. Bound pixel allocation.
With pypdfium2, `page.render(...)` defaults to BGR, and `bitmap.to_numpy()` shares
its buffer. Take `.copy()` before closing the bitmap, page or document. Ensure
three-channel BGR if render options change; do not blindly swap an already-BGR array.
Close resources promptly, including on exceptions, and never materialize an entire
book's rendered pages. Attach zero-based page index + 1 as physical-page provenance.

Use detection max side 960 for the initial fast pass; try 1536 for small text after
visual checks. Higher render dpi alone may not help if detection downsizes again.
The three disabled orientation/unwarping stages suit upright, reasonably flat pages;
verify rotation/skew before relying on that simplification. Enable only necessary
stages with separately verified compatible models, or normalize locally and preserve
the coordinate transform. Do not silently turn them off for unsuitable scans.

## Measure honestly

Warm the persistent instance, separate initialization/render/inference costs, and
compare quality as well as latency on representative content. Check memory and
thermal behavior; tune thread count and batching on the target, not by core-count
assumption. Do not claim energy savings without energy measurement.

`CoreMLExecutionProvider` appearing in a provider list proves neither successful
node placement nor GPU/ANE execution. Require profiling evidence and an end-to-end
comparison before recommending it. Do not assume MLX supports tiny OCR; exclude
archived ports. Official MLX support for a larger vision-language model is a
different workload, not a drop-in default for this detection/recognition pipeline.

API references: [inference engine configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/inference_deployment/local_inference/inference_engine.en.md),
[OCR pipeline and language coverage](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md),
[pypdfium2 Python API](https://pypdfium2.readthedocs.io/en/stable/python_api.html).
