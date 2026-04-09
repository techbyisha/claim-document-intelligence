import logging

from langchain_core.messages import HumanMessage

from utils.llm import get_llm
from utils.parsing import parse_llm_json

logger = logging.getLogger("claims.segregator")

DOCUMENT_TYPES = [
    "claim_forms",
    "cheque_or_bank_details",
    "identity_document",
    "itemized_bill",
    "discharge_summary",
    "prescription",
    "investigation_report",
    "cash_receipt",
    "other",
]

# Sent once per page. Kept tight to avoid token bloat.
CLASSIFICATION_PROMPT = """You are classifying a page from an insurance claim document.

Classify this page into exactly one of these types:
{types}

Use the page image AND the extracted text below to decide.

Extracted text (may be empty for scanned pages):
{text}

Respond ONLY with a JSON object — no explanation, no markdown:
{{"doc_type": "<one of the types above>", "confidence": "high|medium|low", "reason": "<one sentence>"}}"""


def _classify_page(llm, page: dict) -> str:
    prompt_text = CLASSIFICATION_PROMPT.format(
        types=", ".join(DOCUMENT_TYPES),
        text=page["text"][:600] if page["text"] else "(none)",
    )

    message = HumanMessage(content=[
        {"type": "text", "text": prompt_text},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{page['image_b64']}",
                "detail": "low",  # low detail = fewer tokens, sufficient for classification
            },
        },
    ])

    response = llm.invoke([message])
    result = parse_llm_json(response.content, fallback_key="classification_error")

    doc_type = result.get("doc_type", "other")
    if doc_type not in DOCUMENT_TYPES:
        logger.warning(f"Page {page['page_number']}: unrecognised type {doc_type!r}, defaulting to 'other'")
        doc_type = "other"

    logger.info(
        f"Page {page['page_number']}: {doc_type!r} "
        f"(confidence={result.get('confidence', '?')}, reason={result.get('reason', '')!r})"
    )
    return doc_type


def segregator_node(state: dict) -> dict:
    logger.info(f"claim={state['claim_id']!r} — segregating {len(state['pages'])} pages")

    llm = get_llm(temperature=0.0, max_tokens=150)
    classifications: dict[str, list[int]] = {t: [] for t in DOCUMENT_TYPES}

    for page in state["pages"]:
        doc_type = _classify_page(llm, page)
        classifications[doc_type].append(page["page_number"])

    non_empty = {k: v for k, v in classifications.items() if v}
    logger.info(f"claim={state['claim_id']!r} — classification summary: {non_empty}")

    return {"page_classifications": classifications}
