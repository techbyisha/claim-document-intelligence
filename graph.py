import logging

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from agents.bill_agent import bill_agent_node
from agents.discharge_agent import discharge_agent_node
from agents.id_agent import id_agent_node
from agents.segregator import segregator_node

logger = logging.getLogger("claims.graph")


class ClaimState(TypedDict):
    claim_id: str
    pdf_bytes: bytes
    pages: list
    page_classifications: dict
    id_data: dict
    discharge_data: dict
    bill_data: dict
    final_result: dict


def aggregator_node(state: ClaimState) -> dict:
    logger.info(f"claim={state['claim_id']!r} — aggregating results from all agents")

    classifications = state.get("page_classifications", {})
    pages_by_type = {k: v for k, v in classifications.items() if v}

    return {
        "final_result": {
            "claim_id": state["claim_id"],
            "document_map": pages_by_type,
            "total_pages_processed": len(state.get("pages", [])),
            "identity_information": state.get("id_data") or {},
            "discharge_summary": state.get("discharge_data") or {},
            "itemized_bill": state.get("bill_data") or {},
        }
    }


def build_graph():
    workflow = StateGraph(ClaimState)

    workflow.add_node("segregator", segregator_node)
    workflow.add_node("id_agent", id_agent_node)
    workflow.add_node("discharge_agent", discharge_agent_node)
    workflow.add_node("bill_agent", bill_agent_node)
    workflow.add_node("aggregator", aggregator_node)

    workflow.set_entry_point("segregator")

    # segregator fans out to all 3 extraction agents in parallel
    workflow.add_edge("segregator", "id_agent")
    workflow.add_edge("segregator", "discharge_agent")
    workflow.add_edge("segregator", "bill_agent")

    # all agents converge at aggregator
    workflow.add_edge("id_agent", "aggregator")
    workflow.add_edge("discharge_agent", "aggregator")
    workflow.add_edge("bill_agent", "aggregator")

    workflow.add_edge("aggregator", END)

    return workflow.compile()
