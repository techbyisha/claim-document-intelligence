import logging

from langchain_core.messages import HumanMessage

from utils.llm import get_llm
from utils.parsing import parse_llm_json
from utils.pdf_utils import extract_pages_text, pdf_to_page_images

logger = logging.getLogger("claims.bill_agent")

EXTRACTION_PROMPT = """You are extracting billing data from a hospital bill or receipt.

Look carefully at every line item in the document. Extract ALL items — do not skip any.
Do not leave fields as null if the information is visible anywhere in the text or image.

Return ONLY this JSON object (no explanation, no markdown):
{{
  "bill_number": null,
  "bill_date": null,
  "hospital_name": null,
  "patient_name": null,
  "items": [
    {{
      "description": "",
      "category": "room|procedure|pharmacy|investigation|consultation|other",
      "quantity": 1,
      "unit_price": 0.0,
      "total": 0.0
    }}
  ],
  "subtotal": null,
  "tax_amount": null,
  "discount_amount": null,
  "advance_paid": null,
  "grand_total": null,
  "currency": "USD",
  "payment_mode": null
}}

Document text:
{text}"""


def _build_message(page_images: list, text: str) -> HumanMessage:
    content = [{"type": "text", "text": EXTRACTION_PROMPT.format(text=text[:5000])}]
    for img in page_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['image_b64']}",
                "detail": "high",
            }
        })
    return HumanMessage(content=content)


def bill_agent_node(state: dict) -> dict:
    claim_id = state["claim_id"]

    page_nums = (
        state["page_classifications"].get("itemized_bill", [])
        + state["page_classifications"].get("cash_receipt", [])
    )
    page_nums = sorted(set(page_nums))

    if not page_nums:
        logger.info(f"claim={claim_id!r} — no bill pages found, skipping")
        return {"bill_data": {"status": "no bill pages found"}}

    logger.info(f"claim={claim_id!r} — bill agent processing pages {page_nums}")

    text = extract_pages_text(state["pdf_bytes"], page_nums)
    logger.info(f"claim={claim_id!r} — extracted {len(text)} chars from pages {page_nums}")

    all_pages = pdf_to_page_images(state["pdf_bytes"])
    page_images = [p for p in all_pages if p["page_number"] in page_nums]

    llm = get_llm(temperature=0.0, max_tokens=2500)
    response = llm.invoke([_build_message(page_images, text)])

    data = parse_llm_json(response.content, fallback_key="raw_extraction")
    data["source_pages"] = page_nums

    item_count = len(data.get("items") or [])
    logger.info(
        f"claim={claim_id!r} — bill agent done, "
        f"{item_count} line items, total={data.get('grand_total')}"
    )
    return {"bill_data": data}
