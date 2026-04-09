import base64
import logging

import fitz  # pymupdf

logger = logging.getLogger("claims.pdf")

# Render at 1.5x zoom — good enough for vision LLM, keeps image size sane
RENDER_ZOOM = 1.5


def pdf_to_page_images(pdf_bytes: bytes) -> list[dict]:
    """
    Render every page of a PDF as a PNG and return base64-encoded images
    alongside raw extracted text. Both are passed to the segregator so it
    can use whichever gives a clearer signal.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for i in range(len(doc)):
        page = doc[i]
        mat = fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        text = page.get_text().strip()

        pages.append({
            "page_number": i + 1,
            "image_b64": b64,
            "text": text,
            "has_text": len(text) > 30,  # flag scanned pages early
        })

        logger.debug(f"Rendered page {i + 1}/{len(doc)} — {len(img_bytes) // 1024}KB")

    doc.close()
    logger.info(f"Rendered {len(pages)} pages from PDF")
    return pages


def extract_pages_text(pdf_bytes: bytes, page_numbers: list[int]) -> str:
    """
    Pull text from specific page numbers only (1-indexed).
    Agents call this with their assigned pages so they never touch the full PDF.
    """
    if not page_numbers:
        return ""

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    chunks = []

    for pg in sorted(set(page_numbers)):
        if not (1 <= pg <= len(doc)):
            logger.warning(f"Page {pg} out of range (doc has {len(doc)} pages), skipping")
            continue
        text = doc[pg - 1].get_text().strip()
        if text:
            chunks.append(f"[Page {pg}]\n{text}")
        else:
            chunks.append(f"[Page {pg}]\n(no extractable text — likely scanned image)")

    doc.close()
    return "\n\n".join(chunks)
