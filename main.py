from fastapi import Depends, FastAPI, HTTPException, Request, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from app import db
from app.models.pydantic_models import QuestionInput, AnswerResponse, AnswerOutput, ThumbsAction
from app.models.models import Base, DocumentChunk, Feedback
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional, Tuple
import openai
import requests
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.responses import FileResponse
from pathlib import Path
from uuid import uuid4
import re
import logging
from html import escape

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip() or None
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "").strip() or None
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip() or None
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document Intelligence",
    description="Bolt Junk Removal RAG",
    version="1.0.0"
)

# Add CORS middleware for ngrok support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "app/uploads"))
ALLOWED_UPLOAD_TYPES = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@app.on_event("startup")
def initialize_database():
    # Keeps a fresh local install usable without requiring Alembic first.
    Base.metadata.create_all(bind=db.engine)


def _extract_document_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md"}:
        return content.decode("utf-8-sig", errors="replace")
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
            from io import BytesIO
            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}")
    if suffix == ".docx":
        try:
            from docx import Document as WordDocument
            from io import BytesIO
            document = WordDocument(BytesIO(content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read DOCX: {exc}")
    raise HTTPException(status_code=400, detail="Supported formats: PDF, DOCX, TXT, and MD.")


def _split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


@app.post("/upload-document/")
async def upload_document(file: UploadFile = File(...), db_session: Session = Depends(db.get_db_connection)):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in ALLOWED_UPLOAD_TYPES:
        raise HTTPException(status_code=400, detail="Upload a PDF, DOCX, TXT, or MD file.")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is too large (maximum 20 MB).")
    text = _extract_document_text(filename, content)
    chunks = _split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text found in this file.")
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        stored_name = f"{uuid4().hex}_{filename}"
        (UPLOAD_DIR / stored_name).write_bytes(content)
        db_session.add_all([
            DocumentChunk(id=str(uuid4()), source=stored_name, text=chunk, page_number=number)
            for number, chunk in enumerate(chunks, start=1)
        ])
        db_session.commit()
    except SQLAlchemyError as exc:
        db_session.rollback()
        (UPLOAD_DIR / stored_name).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Could not save document text.") from exc
    return {"message": "Document uploaded and indexed.", "filename": filename, "chunks": len(chunks)}

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency can fail in constrained environments
    SentenceTransformer = None


def _get_fallback_answer(question: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "I don't have any relevant information in the uploaded documents to answer this question."

    excerpt = "\n\n".join(chunk[:600] for chunk in context_chunks[:3])
    return (
        "<h2>Relevant text from your document</h2>"
        "<p>The app found relevant passages, but could not generate an AI answer. Check that your Groq API key is valid and that the configured model is available. The passages below are from your uploaded document.</p>"
        f"<p><strong>Question:</strong> {escape(question)}</p>"
        f"<p><strong>Relevant context:</strong></p><p>{excerpt}</p>"
        "<p>Local search is working. Pinecone is optional; Groq must be reachable with a valid API key for generated answers.</p>"
    )


def get_pinecone_index() -> Tuple[Optional[object], Optional[str]]:
    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
        return None, "Pinecone is not configured."

    try:
        client = Pinecone(api_key=PINECONE_API_KEY)
        index_names = client.list_indexes().names()
        if PINECONE_INDEX_NAME not in index_names:
            return None, f"Index '{PINECONE_INDEX_NAME}' was not found in Pinecone."
        return client.Index(PINECONE_INDEX_NAME), None
    except Exception as exc:  # pragma: no cover - depends on external service availability
        return None, str(exc)


def get_sentence_model():
    if SentenceTransformer is None:
        return None

    try:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:  # pragma: no cover - model download can fail when network access is unavailable
        return None


index = None
index_error = None
model = None


def ensure_runtime_services():
    global index, index_error, model
    if index is None and index_error is None:
        index, index_error = get_pinecone_index()
    if model is None and index is not None:
        model = get_sentence_model()
    return index, model

@app.get("/", response_class=HTMLResponse)
def serve_index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    ensure_runtime_services()
    return {
        "status": "healthy",
        "message": "RAG Question Answer API is running",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "pinecone": "ready" if index is not None else "disabled",
            "embedding_model": "ready" if model is not None else "disabled",
            "groq": "configured" if GROQ_API_KEY else "missing",
        },
        "pinecone_error": index_error,
    }


@app.get("/test-pinecone/")
def test_pinecone_connection():
    """Compatibility endpoint for health checks and service verification."""
    ensure_runtime_services()
    if index is not None:
        return {"status": "ok", "message": "Pinecone is configured and reachable."}
    return {"status": "unconfigured", "message": index_error or "Pinecone is not configured."}


# Groq LLM setup
openai.api_key = GROQ_API_KEY
openai.api_base = "https://api.groq.com/openai/v1"


def _groq_requests_session():
    """Use direct internet by default; allow an explicit proxy when required."""
    session = requests.Session()
    session.trust_env = False
    proxy = os.getenv("GROQ_PROXY", "").strip()
    if proxy:
        session.proxies.update({"http": proxy, "https": proxy})
    return session


openai.requestssession = _groq_requests_session


def generate_answer_with_groq(question: str, context_chunks: List[str]) -> str:
    if not context_chunks:
        return "I don't have any relevant information in the uploaded documents to answer this question."

    if not GROQ_API_KEY:
        return _get_fallback_answer(question, context_chunks)

    context = "\n\n".join(context_chunks)
    prompt = f"""You are a world-class strategic advisor for junk removal businesses. You are trained on internal playbooks, SOPs, and performance documents. Your job is to answer questions using ONLY the CONTEXT provided below. All answers must be direct, deeply informed, and packed with real-world examples, industry-specific metrics, and implementation guidance.

### YOUR MISSION:
You are preparing a leadership briefing for a $10M+ junk removal company seeking strategic clarity. Your answers must reflect executive-level insight and draw exclusively from the provided documents.

### RULES (Do NOT break):
1. Read the CONTEXT carefully and extract only the most relevant, actionable insights.
2. Read the QUESTION carefully and ensure your answer is directly relevant to it.
3. Use **specific solid examples from the documents** — include company names, roles, KPIs, events, or initiatives like “505-Junk's 7-year RCMP Toy Drive” or “90%+ waste tech scores”.
4. Tie all content to the **junk removal industry** — highlight sector-specific challenges (e.g., seasonal swings, route optimization, hoarding response, etc.).
5. Synthesize across multiple sections if needed — merge ideas from audit templates, SOPs, strategy guides, etc.
6. Deliver **clear, step-by-step recommendations or behaviors** when action is needed (e.g., time blocking, leadership rhythms, training methods).
7. Avoid any generic statements, summaries, or filler. Stay grounded in the documents.
8. If the question cannot be answered using the context, reply with:
   ❗ “The documents don't contain sufficient information to answer this question.”
9. If the question involves KPIs, budget, SOPs, hours, or time tracking, always:
   - Provide **precise numbers from the context** (e.g., total hours, weekly average)
   - Clearly **show the formula** or calculation steps
   - Recommend **1-2 leadership behaviors** tied to the metric
   - Propose a **SOP or system change** if improvement is needed
   - End with a concise, confident **final answer** (e.g., "The final weekly average is 320.83 hours.")
10. If the question involves productivity, deep work, or time audits:
   - Identify **energy rating** per task (Energizing, Neutral, Draining) from context.
   - Extract and correlate **task value ($ to $$$$)** and **achieved outcomes (e.g., GP, hours logged)**.
   - Detect trends: Which tasks are energizing? Which are draining?
   - Apply the **Prioritization Framework**:
     - Schedule Energizing + High-Value tasks during peak energy hours.
     - Batch Draining or Neutral tasks in low-focus slots.
   - Recommend time blocking, energy recovery strategies, or SOP adjustments.
   - Final output should state: 
     - The trend in energy ratings
     - A specific and solid example
     - Optimization recommendations

11. If referencing any structured framework (e.g., GSR Meeting, Vivid Vision, Accountability Chart), output the FULL structure with labeled sections and implementation detail.

12. If the content touches Vivid Vision:
   - Include complete sections: Culture, Metrics, Market Position, Systems, People, Leadership.
   - Use sensory-rich and specific examples from the docs.
   - Follow 3-Year Vision/Painted Picture formats if provided.

13. For every procedural answer, label who owns what:
   - Use “Leadership Responsibility” for strategic roles.
   - Use “Team Contributor Task” for executional duties.
   - Resolve role ambiguity using SOPs or role charts in context.

14. Use structured formatting when showing frameworks or templates:
   - Use tables, headings, or indented bullets.
   - Examples: GSR Meetings, SOP steps, Weekly Rhythm, Accountability Charts, KPI Reports.

### CONTEXT:
{context}

### QUESTION:
{question}

### FINAL ANSWER:
Write like you're briefing the CEO. Include quotes, numbers, exact SOPs, or leadership behaviors from the documents when possible. Be crisp, credible, and correct.
return the output in html format.
Convert the following text into clean, properly structured HTML.

Please follow these strict formatting rules:

1. Use <h1>, <h2>, <h3>, etc., for any line that starts with #, ##, ###, etc.
2. Use <strong>...</strong> for any text surrounded by one or more asterisks (*) on both sides — including *, **, or ***.
3. Use <ol> and <li> tags for all numbered lists (lines starting with "1.", "2.", etc.). DO NOT use <ul> under any condition.
4. Wrap all regular text (non-heading, non-list) in <p> tags to ensure readability and structure.
5. Make the output clean, properly indented, and visually presentable as raw HTML.
6. Maintain original content meaning and section groupings.

Ensure the final result is in pure HTML format and follows all the above instructions precisely.

"""

    try:
        response = openai.ChatCompletion.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a senior strategy analyst who extracts operational insights from internal documents. Prioritize precision, synthesis, industry relevance, and specific examples."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            seed=42,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            max_tokens=3000
        )
        answer = response["choices"][0]["message"]["content"].strip()
        answer = re.sub(r"^```(?:html)?\s*|\s*```$", "", answer, flags=re.IGNORECASE).strip()
        return answer
    except Exception:
        logger.exception("Groq answer generation failed (model=%s)", GROQ_MODEL)
        return _get_fallback_answer(question, context_chunks)


def _fallback_document_search(question: str, db_session: Session, limit: int = 5):
    question_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
    if not question_tokens:
        return []

    chunks = db_session.query(DocumentChunk).all()
    scored = []
    stop_words = {"the", "and", "for", "with", "from", "what", "when", "where", "which", "who", "how", "does", "this", "that", "are", "was", "were", "have", "has", "about"}
    question_tokens = {token for token in question_tokens if token not in stop_words}
    for chunk in chunks:
        chunk_tokens = set(re.findall(r"[a-z0-9]+", (chunk.text or "").lower()))
        score = len(question_tokens & chunk_tokens)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]


@app.post("/ask-question/", response_model=AnswerResponse)
def ask_question(input_data: QuestionInput, db_session: Session = Depends(db.get_db_connection)):
    try:
        ensure_runtime_services()
        generic_inputs = {"hi", "hello", "hey", "how are you", "who are you", "what can you do"}
        lower_q = input_data.question.lower().strip()
        if lower_q in generic_inputs:
            return AnswerResponse(
                question=input_data.question,
                final_answer="Hi, I am an AI assistant from Bolt Junk Removal. How can I assist you today?",
                answers=[],
                source_document="System Response"
            )

        matched_ids = []
        if index is not None and model is not None:
            try:
                query_embedding = model.encode(input_data.question).tolist()
                search_result = index.query(vector=query_embedding, top_k=25, include_metadata=False)
                matches = getattr(search_result, "matches", None)
                if matches is None:
                    matches = search_result.get("matches", [])
                matched_ids = [getattr(match, "id", None) or match.get("id") for match in matches]
                matched_ids = [item for item in matched_ids if item]
            except Exception:
                # A Pinecone outage should not disable local document search.
                matched_ids = []

        if matched_ids:
            rows = db_session.query(DocumentChunk).filter(DocumentChunk.id.in_(matched_ids)).all()
            by_id = {row.id: row for row in rows}
            results = [by_id[item] for item in matched_ids if item in by_id]
        else:
            results = _fallback_document_search(input_data.question, db_session)

        if not results:
            raise HTTPException(status_code=404, detail="No matching documents found.")

        answers = []
        chunks = []
        source_documents = set()

        for chunk_row in results:
            source_documents.add(chunk_row.source)
            answers.append(AnswerOutput(
                content=chunk_row.text,
                source=chunk_row.source,
                source_document=chunk_row.source,
                score=0.0
            ))
            chunks.append(chunk_row.text)

        all_source_documents = sorted(list(source_documents))
        final_answer = generate_answer_with_groq(input_data.question, chunks)
        return AnswerResponse(
            question=input_data.question,
            final_answer=final_answer,
            answers=answers,
            source_document=f"Sources: {', '.join(all_source_documents[:4])}"
        )

    except HTTPException as he:
        raise he
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}")

@app.get("/download/{source_name}")
async def download_file(source_name: str):
    """Download file by searching in uploads directory"""
    upload_dir = os.getenv("UPLOAD_DIR", "app/uploads")
    
    # Search for file with this source name
    possible_extensions = ['.pdf', '.docx', '.txt', '.md']
    
    for ext in possible_extensions:
        safe_name = Path(source_name).name
        if Path(safe_name).suffix.lower() != ext:
            continue
        file_path = Path(upload_dir) / safe_name
        if file_path.is_file():
            return FileResponse(path=file_path, filename=safe_name, media_type='application/octet-stream')
    
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/feedback/")
async def feedback(payload: ThumbsAction,db: Session = Depends(db.get_db_connection)):
    comment = payload.comment or ""
    if comment == "":
        status = True
    else:
        status = False
      

    feedback_entry = Feedback(
        question=payload.question,
        answer=payload.answer,
        status=status,
        comments=comment,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

    try:
        db.add(feedback_entry)
        db.commit()
        db.refresh(feedback_entry)
    except SQLAlchemyError as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "message": "Feedback received",
        "data": {
            "id": feedback_entry.id,
            "question": feedback_entry.question,
            "answer": feedback_entry.answer,
            "status": feedback_entry.status,
            "comments": feedback_entry.comments,
            "created_at": feedback_entry.created_at,
            "updated_at": feedback_entry.updated_at,
        }
    }


@app.get("/feedbacks/")
async def get_feedbacks(status: Optional[bool] = Query(None), db: Session = Depends(db.get_db_connection)):
    """
    Fetch feedback entries optionally filtered by status.
    - If `status` query param is provided (true or false), filter results by status.
    - Otherwise, return all feedback entries.
    """
    query = db.query(Feedback)

    if status is not None:
        query = query.filter(Feedback.status == status)

    feedbacks = query.all()

    # Prepare a list of responses, excluding or including fields as needed.
    results = []
    for fb in feedbacks:
        results.append({
            "id": fb.id,
            "question": fb.question,
            "answer": fb.answer,
            "status": fb.status,
            "comments": fb.comments,
            "created_at": fb.created_at,
            "updated_at": fb.updated_at,
        })

    return {
        "count": len(results),
        "data": results
    }
