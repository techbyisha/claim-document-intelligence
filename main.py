from dotenv import load_dotenv
load_dotenv()

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from graph import build_graph
from utils.pdf_utils import pdf_to_page_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("claims.api")

claim_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global claim_graph
    logger.info("Compiling LangGraph workflow...")
    claim_graph = build_graph()
    logger.info("Graph ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Claim Document Intelligence API",
    description="LangGraph-powered multi-agent pipeline for insurance claim extraction.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_PDF_SIZE_MB = 20


@app.middleware("http")
async def log_requests(request: Request, call_next):
    req_id = str(uuid.uuid4())[:8]
    logger.info(f"[{req_id}] {request.method} {request.url.path}")
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed = round((time.perf_counter() - t0) * 1000)
    logger.info(f"[{req_id}] → {response.status_code} in {elapsed}ms")
    return response


@app.post("/api/process", summary="Process a claim PDF")
async def process_claim(
    claim_id: str = Form(..., description="Unique claim identifier"),
    file: UploadFile = File(..., description="PDF file of the claim"),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are accepted.")

    pdf_bytes = await file.read()

    size_mb = len(pdf_bytes) / (1024 * 1024)
    if size_mb > MAX_PDF_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f}MB). Limit is {MAX_PDF_SIZE_MB}MB.",
        )

    if len(pdf_bytes) < 100:
        raise HTTPException(status_code=400, detail="File appears to be empty or corrupt.")

    logger.info(f"Processing claim={claim_id!r}, file={file.filename!r}, size={size_mb:.2f}MB")

    try:
        pages = pdf_to_page_images(pdf_bytes)
    except Exception as e:
        logger.error(f"PDF rendering failed for claim={claim_id!r}: {e}")
        raise HTTPException(status_code=422, detail=f"Could not render PDF: {str(e)}")

    logger.info(f"claim={claim_id!r} — {len(pages)} pages detected, starting graph...")

    initial_state = {
        "claim_id": claim_id,
        "pdf_bytes": pdf_bytes,
        "pages": pages,
        "page_classifications": {},
        "id_data": {},
        "discharge_data": {},
        "bill_data": {},
        "final_result": {},
    }

    try:
        result = await claim_graph.ainvoke(initial_state)
    except Exception as e:
        logger.exception(f"Graph execution failed for claim={claim_id!r}: {e}")
        raise HTTPException(status_code=500, detail="Pipeline execution failed. Check server logs.")

    logger.info(f"claim={claim_id!r} — processing complete.")
    return JSONResponse(content=result["final_result"])


@app.get("/health", include_in_schema=False)
async def health():
    return {"status": "ok", "graph_ready": claim_graph is not None}


@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Claim Document Intelligence API is running. POST to /api/process to submit a claim."}
