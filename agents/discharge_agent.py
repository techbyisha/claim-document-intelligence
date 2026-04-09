import logging

from langchain_core.messages import HumanMessage

from utils.llm import get_llm
from utils.parsing import parse_llm_json
from utils.pdf_utils import extract_pages_text, pdf_to_page_images

logger = logging.getLogger("claims.discharge_agent")

EXTRACTION_PROMPT = """You are extracting clinical data from a hospital discharge summary.

Look carefully at the document and extract every clinical detail visible.
Do not leave anything as null if the information is present anywhere.

Return ONLY this JSON object (no explanation, no markdown):
{{
  "hospital_name": null,
  "admission_date": null,
  "discharge_date": null,
  "length_of_stay_days": null,
  "ward_or_room_type": null,
  "treating_physician": null,
  "primary_diagnosis": null,
  "secondary_diagnoses": [],
  "icd_codes": [],
  "procedures_performed": [],
  "discharge_condition": null,
  "follow_up_instructions": null,
  "referred_by": null
}}

Document text:
{text}"""


def _build_message(page_images: list, text: str) -> HumanMessage:
    content = [{"type": "text", "text": EXTRACTION_PROMPT.format(text=text[:4000])}]
    for img in page_images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['image_b64']}",
                "detail": "high",
            }
        })
    return HumanMessage(content=content)


def discharge_agent_node(state: dict) -> dict:
    claim_id = state["claim_id"]
    page_nums = state["page_classifications"].get("discharge_summary", [])

    if not page_nums:
        logger.info(f"claim={claim_id!r} — no discharge_summary pages, skipping")
        return {"discharge_data": {"status": "no discharge summary pages found"}}

    logger.info(f"claim={claim_id!r} — discharge agent processing pages {page_nums}")

    text = extract_pages_text(state["pdf_bytes"], page_nums)
    logger.info(f"claim={claim_id!r} — extracted {len(text)} chars from pages {page_nums}")

    all_pages = pdf_to_page_images(state["pdf_bytes"])
    page_images = [p for p in all_pages if p["page_number"] in page_nums]

    llm = get_llm(temperature=0.0, max_tokens=1200)
    response = llm.invoke([_build_message(page_images, text)])

    data = parse_llm_json(response.content, fallback_key="raw_extraction")
    data["source_pages"] = page_nums

    logger.info(
        f"claim={claim_id!r} — discharge agent done, "
        f"diagnosis={data.get('primary_diagnosis')!r}, "
        f"stay={data.get('length_of_stay_days')} days"
    )
    return {"discharge_data": data}
