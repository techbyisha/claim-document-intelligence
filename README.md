# Claims Processor

A FastAPI service that extracts structured data from insurance claim PDFs using a multi-agent LangGraph pipeline powered by GPT-4o vision.

Each page of the PDF is classified by document type, then specialist agents extract relevant information in parallel — returning a single clean JSON response.

---

## Table of Contents

- [How It Works](#how-it-works)
- [Setup](#setup)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
  - [POST /api/process](#post-apiprocess)
  - [GET /health](#get-health)
- [Input](#input)
- [Output](#output)
  - [document\_map](#document_map)
  - [identity\_information](#identity_information)
  - [discharge\_summary](#discharge_summary)
  - [itemized\_bill](#itemized_bill)
- [Document Types Recognized](#document-types-recognized)
- [Error Responses](#error-responses)
- [Processing Pipeline](#processing-pipeline)
- [Deployment](#deployment)

---

## How It Works

```
PDF + Claim ID
      │
      ▼
┌─────────────────┐
│   SEGREGATOR    │  ← GPT-4o vision classifies every page into 1 of 9 document types
└────────┬────────┘
         │
    ┌────┴────┐────────────┐
    ▼         ▼            ▼
┌────────┐ ┌──────────┐ ┌────────┐
│   ID   │ │DISCHARGE │ │  BILL  │  ← Three agents run in parallel
│ AGENT  │ │  AGENT   │ │ AGENT  │     Each only sees its relevant pages
└────┬───┘ └────┬─────┘ └───┬────┘
     │          │            │
     └──────────┼────────────┘
                ▼
         ┌────────────┐
         │ AGGREGATOR │  ← Merges results into a single JSON response
         └────────────┘
```

**Key design**: Each specialist agent only receives the pages relevant to it — not the full PDF. This minimizes token usage and improves accuracy.

---

## Setup

**Requirements**: Python 3.9+, an OpenAI API key with GPT-4o access.

```bash
cd claims-processor

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure your API key
# Open .env and set: OPENAI_API_KEY=sk-...
```

---

## Running the Server

```bash
uvicorn main:app --reload
```

The server starts at `http://localhost:8000`.

- Interactive API docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

**Production:**
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Run the test script against a local server:**
```bash
python tests/test_local.py /path/to/claim.pdf
```

---

## API Reference

### POST /api/process

Processes a claim PDF and returns extracted structured data.

**Request** — `multipart/form-data`

| Field      | Type   | Required | Description                         |
|------------|--------|----------|-------------------------------------|
| `claim_id` | string | Yes      | Unique identifier for the claim     |
| `file`     | file   | Yes      | PDF file containing claim documents |

**Constraints:**
- File must be a valid `.pdf`
- Maximum size: 20 MB
- Minimum size: 100 bytes (rejects empty/corrupt files)

**Example:**
```bash
curl -X POST http://localhost:8000/api/process \
  -F "claim_id=CLM-2024-001" \
  -F "file=@/path/to/claim.pdf"
```

---

### GET /health

Returns server and pipeline status.

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "graph_ready": true
}
```

---

## Input

The service accepts a single multipart form submission:

| Field      | Example Value     | Notes                                       |
|------------|-------------------|---------------------------------------------|
| `claim_id` | `"CLM-2024-001"`  | Any string; echoed back in the response     |
| `file`     | `claim.pdf`       | Must be a real PDF; scanned PDFs supported  |

The PDF can contain any mix of pages — claim forms, ID cards, hospital records, bills, receipts, prescriptions, lab reports, etc. Page order does not matter.

---

## Output

On success, the API returns HTTP `200` with a JSON body:

```json
{
  "claim_id": "CLM-2024-789456",
  "total_pages_processed": 18,
  "document_map": { ... },
  "identity_information": { ... },
  "discharge_summary": { ... },
  "itemized_bill": { ... }
}
```

---

### document_map

A map of every document type to the page numbers (1-indexed) classified as that type.

```json
"document_map": {
  "claim_forms":            [1, 13],
  "identity_document":      [3],
  "discharge_summary":      [4],
  "itemized_bill":          [9, 10],
  "cash_receipt":           [7],
  "prescription":           [5],
  "investigation_report":   [6, 11, 12, 17],
  "cheque_or_bank_details": [2],
  "other":                  [8, 14, 15, 16, 18]
}
```

Pages that don't match any known type appear under `"other"`. Empty arrays mean no pages of that type were found.

---

### identity_information

Extracted from pages classified as `identity_document` and `claim_forms`.

```json
"identity_information": {
  "patient_name":        "John Michael Smith",
  "date_of_birth":       "1978-06-14",
  "gender":              "Male",
  "id_type":             "Government ID",
  "id_number":           "DL-4821-9930",
  "policy_number":       "POL-CLM-2024-789456",
  "insurance_provider":  "BlueCross BlueShield",
  "member_id":           "MEM-789456",
  "address":             "4821 Maple Street, Springfield, IL 62701",
  "contact_number":      "+1 (217) 555-0192",
  "source_pages":        [1, 3, 13]
}
```

| Field                | Type           | Description                                     |
|----------------------|----------------|-------------------------------------------------|
| `patient_name`       | string or null | Full name of the patient                        |
| `date_of_birth`      | string or null | Date of birth (format as found in document)     |
| `gender`             | string or null | Gender                                          |
| `id_type`            | string or null | Type of ID (Aadhaar, PAN, Passport, etc.)       |
| `id_number`          | string or null | ID document number                              |
| `policy_number`      | string or null | Insurance policy number                         |
| `insurance_provider` | string or null | Name of the insurance company                   |
| `member_id`          | string or null | Insurance member/beneficiary ID                 |
| `address`            | string or null | Residential address                             |
| `contact_number`     | string or null | Phone number                                    |
| `source_pages`       | array[int]     | Page numbers this data was extracted from       |

If no identity pages are found in the PDF:
```json
"identity_information": {
  "status": "no identity pages found"
}
```

---

### discharge_summary

Extracted from pages classified as `discharge_summary`.

```json
"discharge_summary": {
  "hospital_name":          "Springfield General Hospital",
  "admission_date":         "2024-03-08",
  "discharge_date":         "2024-03-13",
  "length_of_stay_days":    5,
  "ward_or_room_type":      "Medical Ward",
  "treating_physician":     "Dr. Patricia Nguyen",
  "primary_diagnosis":      "Community Acquired Pneumonia (CAP)",
  "secondary_diagnoses":    ["Type 2 Diabetes Mellitus", "Hypertension"],
  "icd_codes":              ["J18.9", "E11.9", "I10"],
  "procedures_performed":   ["IV Antibiotic Therapy", "Chest X-Ray", "Sputum Culture"],
  "discharge_condition":    "Stable, improved",
  "follow_up_instructions": "Follow up with PCP in 7 days, complete antibiotic course",
  "referred_by":            "Dr. Alan Torres",
  "source_pages":           [4]
}
```

| Field                    | Type            | Description                                         |
|--------------------------|-----------------|-----------------------------------------------------|
| `hospital_name`          | string or null  | Name of the treating hospital                       |
| `admission_date`         | string or null  | Date of admission                                   |
| `discharge_date`         | string or null  | Date of discharge                                   |
| `length_of_stay_days`    | integer or null | Total days of hospitalisation                       |
| `ward_or_room_type`      | string or null  | Room category (e.g., ICU, Private, General)         |
| `treating_physician`     | string or null  | Name of the primary doctor                          |
| `primary_diagnosis`      | string or null  | Main diagnosis                                      |
| `secondary_diagnoses`    | array[string]   | Additional diagnoses                                |
| `icd_codes`              | array[string]   | ICD-10 diagnosis codes                              |
| `procedures_performed`   | array[string]   | Surgical or clinical procedures performed           |
| `discharge_condition`    | string or null  | Patient condition at discharge                      |
| `follow_up_instructions` | string or null  | Post-discharge care instructions                    |
| `referred_by`            | string or null  | Referring physician (if any)                        |
| `source_pages`           | array[int]      | Page numbers this data was extracted from           |

If no discharge summary pages are found:
```json
"discharge_summary": {
  "status": "no discharge summary pages found"
}
```

---

### itemized_bill

Extracted from pages classified as `itemized_bill` and `cash_receipt`.

```json
"itemized_bill": {
  "bill_number":      "BILL-20240313-7894",
  "bill_date":        "2024-03-13",
  "hospital_name":    "Springfield General Hospital",
  "patient_name":     "John Michael Smith",
  "items": [
    {
      "description": "Semi-Private Room (5 nights)",
      "category":    "room",
      "quantity":    5,
      "unit_price":  650.00,
      "total":       3250.00
    },
    {
      "description": "Piperacillin-Tazobactam 4.5g IV",
      "category":    "pharmacy",
      "quantity":    12,
      "unit_price":  38.75,
      "total":       465.00
    },
    {
      "description": "Chest X-Ray (PA View)",
      "category":    "investigation",
      "quantity":    2,
      "unit_price":  185.00,
      "total":       370.00
    },
    {
      "description": "Pulmonology Consultation",
      "category":    "consultation",
      "quantity":    1,
      "unit_price":  320.00,
      "total":       320.00
    }
    // ... 16 more line items
  ],
  "subtotal":        6218.65,
  "tax_amount":      0.00,
  "discount_amount": 0.00,
  "advance_paid":    200.00,
  "grand_total":     6418.65,
  "currency":        "USD",
  "payment_mode":    "Insurance",
  "source_pages":    [7, 9, 10]
}
```

**Top-level bill fields:**

| Field             | Type           | Description                                 |
|-------------------|----------------|---------------------------------------------|
| `bill_number`     | string or null | Hospital bill / invoice number              |
| `bill_date`       | string or null | Date the bill was generated                 |
| `hospital_name`   | string or null | Name of the billing hospital                |
| `patient_name`    | string or null | Patient name on the bill                    |
| `subtotal`        | float or null  | Sum before tax/discount                     |
| `tax_amount`      | float or null  | Tax applied                                 |
| `discount_amount` | float or null  | Discount applied                            |
| `advance_paid`    | float or null  | Amount already paid in advance              |
| `grand_total`     | float or null  | Final amount due                            |
| `currency`        | string         | Currency code (e.g., `INR`, `USD`)          |
| `payment_mode`    | string or null | Mode of payment                             |
| `source_pages`    | array[int]     | Page numbers this data was extracted from   |

**Each item in `items`:**

| Field         | Type    | Description                                                                         |
|---------------|---------|-------------------------------------------------------------------------------------|
| `description` | string  | Name/description of the charge                                                      |
| `category`    | string  | One of: `room`, `procedure`, `pharmacy`, `investigation`, `consultation`, `other`   |
| `quantity`    | integer | Quantity billed                                                                     |
| `unit_price`  | float   | Price per unit                                                                      |
| `total`       | float   | `quantity × unit_price`                                                             |

If no bill pages are found:
```json
"itemized_bill": {
  "status": "no itemized bill pages found"
}
```

---

## Document Types Recognized

The segregator classifies every page into exactly one of these types:

| Type                     | Description                                              |
|--------------------------|----------------------------------------------------------|
| `claim_forms`            | Insurance claim application or pre-authorization forms   |
| `identity_document`      | Government ID, insurance card, Aadhaar, PAN, passport    |
| `discharge_summary`      | Hospital discharge notes and clinical summary            |
| `itemized_bill`          | Hospital bills with line-item charges                    |
| `cash_receipt`           | Payment receipts                                         |
| `prescription`           | Doctor-issued prescriptions                              |
| `investigation_report`   | Lab reports, scan results, pathology reports             |
| `cheque_or_bank_details` | Cheques, bank account or NEFT/RTGS details               |
| `other`                  | Any page that does not match the above types             |

> **Note**: `prescription`, `investigation_report`, and `cheque_or_bank_details` pages are classified and appear in `document_map`, but no specialist extraction agent currently processes them. Their page numbers are recorded but their content is not extracted.

---

## Error Responses

| HTTP Status | Condition                                                    |
|-------------|--------------------------------------------------------------|
| `400`       | File is empty or corrupt (< 100 bytes)                       |
| `413`       | File exceeds the 20 MB size limit                            |
| `415`       | File is not a PDF                                            |
| `422`       | PDF could not be rendered (malformed or password-protected)  |
| `500`       | Internal pipeline error                                      |

All errors follow this format:
```json
{
  "detail": "File must be a PDF"
}
```

---

## Processing Pipeline

The pipeline is implemented as a LangGraph directed graph with these nodes:

| Node             | Model  | Vision Detail | Max Tokens | Purpose                             |
|------------------|--------|---------------|------------|-------------------------------------|
| Segregator       | GPT-4o | low           | 150/page   | Classify each page by document type |
| ID Agent         | GPT-4o | high          | 800        | Extract identity fields             |
| Discharge Agent  | GPT-4o | high          | 1200       | Extract clinical/hospital fields    |
| Bill Agent       | GPT-4o | high          | 2500       | Extract billing line items          |
| Aggregator       | —      | —             | —          | Merge all agent outputs             |

- All three extraction agents run **in parallel** after segregation.
- Temperature is `0.0` on all nodes for deterministic output.
- PDF pages are rendered to PNG at **1.5× zoom** before being passed to the vision model.
- Segregator uses `detail: low` (fast classification); extraction agents use `detail: high` (accurate reading).

---

## Environment Variables

| Variable         | Required | Description                            |
|------------------|----------|----------------------------------------|
| `OPENAI_API_KEY` | Yes      | OpenAI API key with GPT-4o access      |

---

## Deployment

**Render (recommended for quick deploys):**

1. Push this repo to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable: `OPENAI_API_KEY=sk-...`
