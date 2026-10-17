# Ebook and DjVu routing

## EPUB: preserve the original structure

1. Read the EPUB archive locally without running scripts or loading remote content.
   Bound archive expansion; reject member paths escaping the chosen output directory.
   Parse XML without external entity resolution.
2. Read `META-INF/container.xml` to locate the package document. Resolve its manifest
   resource hrefs relative to that document, then follow spine `itemref`/`idref`
   order. ZIP order and alphabetical filenames are not reading order.
3. Parse referenced XHTML, retaining headings, lists, section IDs, links and image
   references. Preserve spine href plus fragment ID as provenance. Handle non-linear
   resources such as notes explicitly rather than silently losing them.
4. Keep existing page-list anchors if present, but do not invent printed page numbers
   from reflowable content. Preserve source-to-output anchors when normalizing text.
5. OCR text-bearing images only when no usable equivalent text exists. Keep the image
   reference and section association; deduplicate against nearby native text.

If a maintained converter is needed, Calibre's OEB output retains HTML resources,
whereas plain TXT discards structure. Conversion can rewrite IDs and layout:
never promise that original anchors survive. Prefer direct EPUB parsing when
stable original section IDs matter.

## MOBI and AZW3: convert, then extract

Calibre documents MOBI (Mobi6/KF8) and AZW3 input and EPUB/OEB output. Verify the
installed version and `ebook-convert` availability before use; it may not be on
PATH even if the desktop application exists. Check official current maintenance
and platform support before proposing installation; do not install automatically.

The following syntax is verified against official documentation, not a guarantee
of successful conversion for a particular file. Choose destinations that do not
exist and inspect output structure after conversion:

```sh
ebook-convert "./input.mobi" "./converted.epub"
ebook-convert "./input.azw3" "./converted-azw3.epub"
ebook-convert "./input.epub" "./converted-oeb"
```

The extensionless final destination requests an OEB directory, not one text file.
After conversion, use the EPUB procedure or parse the OEB HTML in package order.
Record both the original document reference and converted resource hrefs. Do not
convert to PDF just to run OCR. For `.azw`, unknown containers or protected inputs,
identify the actual format and request an authorized accessible copy when needed;
never treat changing a suffix as a decoder or decryption step.

## DjVu: text-bearing and image-only are different cases

Calibre explicitly supports DjVu input **only with embedded text**. With a verified
local installation, use that text route and check output against visible pages;
previous hidden OCR can be inaccurate. Preserve original page mapping where the
conversion exposes it; if it loses that mapping, report the limitation rather than
inventing provenance or claiming a page-faithful result.

DjVuLibre documents `djvutxt` for hidden text/positions and `ddjvu` for rendering,
but its current maintenance is UNVERIFIED here. Under a maintained-tools-only
constraint, do not recommend installing or using it without separately establishing
current maintenance. Documentation availability alone does not establish maintenance.
Do not assume `ddjvu` emits PNG; its documented formats include PPM, an OCR input.

For image-only DjVu, verify a maintained decoder and its actual CLI/API before
proceeding, or ask the user for an accessible exported PDF/images. Calibre's
text-bearing support is not evidence of image-only decoding. Never present an
unverified decoder as an endorsed fallback or patch an upstream dependency.

## Evidence and re-check points

Native OCR routing was checked in PaddleOCR's `paddleocr/_pipelines/ocr.py` and
PaddleX 3.7's `inference/common/batch_sampler/image_batch_sampler.py`: supported
raster suffixes include PNG, JPEG, TIFF, BMP, WEBP and PPM; PDF is rendered, not
parsed for native text by the OCR path. Unsupported formats can yield empty output,
so test decoding and coverage rather than treating no exception as success.

Official references (re-check against the installed version):

- [Calibre supported formats](https://manual.calibre-ebook.com/faq.html#what-formats-does-calibre-support-conversion-to-from)
- [Calibre conversion CLI](https://manual.calibre-ebook.com/generated/en/ebook-convert.html)
- [Calibre release history](https://calibre-ebook.com/whats-new): evidence of active maintenance, not a permanent guarantee.
- [EPUB spine specification](https://www.w3.org/TR/epub-33/#sec-spine-elem)
- [DjVuLibre text command documentation](https://github.com/DjVuLibre/djvulibre/blob/master/tools/djvutxt.1) and [render command documentation](https://github.com/DjVuLibre/djvulibre/blob/master/tools/ddjvu.1): syntax evidence only.
