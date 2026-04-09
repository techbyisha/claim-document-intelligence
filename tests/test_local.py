"""
Quick smoke test — run this before demoing.
Creates a minimal PDF and hits the local API.

Usage:
    python tests/test_local.py
"""

import os
import sys
import requests

API_URL = "http://localhost:8000"
SAMPLE_PDF = os.path.join(os.path.dirname(__file__), "sample.pdf")


def test_health():
    r = requests.get(f"{API_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.text}"
    print(f"  health: {r.json()}")


def test_process(pdf_path: str, claim_id: str = "TEST-001"):
    if not os.path.exists(pdf_path):
        print(f"  No PDF found at {pdf_path}, skipping process test.")
        return

    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{API_URL}/api/process",
            data={"claim_id": claim_id},
            files={"file": ("claim.pdf", f, "application/pdf")},
        )

    if r.status_code == 200:
        result = r.json()
        print(f"  claim_id: {result.get('claim_id')}")
        print(f"  pages_processed: {result.get('total_pages_processed')}")
        print(f"  document_map: {result.get('document_map')}")
        print(f"  patient_name: {result.get('identity_information', {}).get('patient_name')}")
        print(f"  grand_total: {result.get('itemized_bill', {}).get('grand_total')}")
    else:
        print(f"  FAILED {r.status_code}: {r.text}")


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else SAMPLE_PDF

    print("\n--- health ---")
    test_health()

    print("\n--- process ---")
    test_process(pdf_path)
