import logging

from langchain_core.messages import HumanMessage

from utils.llm import get_llm
from utils.parsing import parse_llm_json
from utils.pdf_utils import extract_pages_text, pdf_to_page_images

logger = logging.getLogger("claims.id_agent")

EXTRACTION_PROMPT = """You are extracting data from an insurance claim document.

Look carefully at the document content below and extract every piece of identity
and insurance information you can find. Do not leave anything as null if the information
is visible anywhere in the text or image.

Return ONLY this JSON object (no explanation, no markdown):
{{
  "patient_name": null,
  "date_of_birth": null,
  "gender": null,
  "id_type": null,
  "id_number": null,
  "policy_number": null,
  "insurance_provider": null,
  "member_id": null,
  "address": null,
  "contact_number": null
}}

Document text:
{text}"""


def _build_message(page_images: list, text: str) -> HumanMessage:
    content = [{"type": "text", "text": EXTRACTION_PROMPT.format(text=text[:3000])}]
    for img in page_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['image_b64']}",
                "detail": "high",
            }
        })
    return HumanMessage(content=content)


def id_agent_node(state: dict) -> dict:
    claim_id = state["claim_id"]

    # identity_document has the ID card (name, DOB, address)
    # claim_forms has policy number, insurance provider, contact
    # both are needed for a complete identity extraction
    page_nums = sorted(set(
        state["page_classifications"].get("identity_document", [])
        + state["page_classifications"].get("claim_forms", [])
    ))

    if not page_nums:
        logger.info(f"claim={claim_id!r} — no identity pages found, skipping")
        return {"id_data": {"status": "no identity pages found"}}

    logger.info(f"claim={claim_id!r} — ID agent processing pages {page_nums}")

    text = extract_pages_text(state["pdf_bytes"], page_nums)
    logger.info(f"claim={claim_id!r} — extracted {len(text)} chars from pages {page_nums}")

    all_pages = pdf_to_page_images(state["pdf_bytes"])
    page_images = [p for p in all_pages if p["page_number"] in page_nums]

    llm = get_llm(temperature=0.0, max_tokens=800)
    response = llm.invoke([_build_message(page_images, text)])

    data = parse_llm_json(response.content, fallback_key="raw_extraction")
    data["source_pages"] = page_nums

    logger.info(f"claim={claim_id!r} — ID agent done, patient={data.get('patient_name')!r}")
    return {"id_data": data}
