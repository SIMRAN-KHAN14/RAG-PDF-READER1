# 📄 RAG Question-Answer System

A FastAPI-based Retrieval-Augmented Generation (RAG) application that allows users to upload documents (PDF, DOCX, TXT, Markdown) and ask intelligent, context-aware questions through a clean web interface.

Designed for flexibility, the system seamlessly integrates **Pinecone** and **Groq (Llama)** for cloud-powered vector search and LLM generation, while automatically falling back to **SQLite** and **local document matching** if external API credentials are omitted—ensuring zero-crash local development.

---

## ✨ Key Features

- **Multi-Format Document Ingestion**: Upload and parse `.pdf`, `.docx`, `.txt`, and `.md` files directly from the UI.
- **Hybrid / Resilient Architecture**: Runs out-of-the-box locally with SQLite and local text search; upgrades automatically to PostgreSQL, Pinecone, and Groq when configured.
- **Interactive Web Interface**: Built-in chat UI to upload documents and query them in real time.
- **Production-Ready Deployment**: Includes Docker, Docker Compose, and SSL/HTTPS configuration scripts.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **Database**: SQLite (local default) / PostgreSQL (production)
- **Vector Store**: Pinecone (`384` dimensions, `cosine` metric)
- **LLM Inference**: Groq API
- **Embeddings**: Sentence Transformers
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Deployment**: Docker & Docker Compose

---

## 📋 Prerequisites

- **Python 3.8+**
- *(Optional for Cloud/Production)*:
  - PostgreSQL database
  - [Pinecone](https://www.pinecone.io/) account & API key
  - [Groq](https://console.groq.com/) API key
  - Docker & Docker Compose (for containerized deployment)

---

## 🚀 Quick Start (Local Mode)

You can run the application locally in under two minutes using SQLite without needing Pinecone or Groq API keys.

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/RAG-PDF-READER1.git
cd RAG-PDF-READER1
```

### 2. Create & Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Start the Server
```bash
uvicorn main:app --reload
```

### 5. Ask Questions
1. Open **http://127.0.0.1:8000** in your browser.
2. At the top of the chat page, select a **PDF, DOCX, TXT, or Markdown** file and click **Upload**. *(Note: Scanned PDFs require OCR preprocessing before text can be extracted).*
3. Once uploaded, type your question in the chat box at the bottom and press **Enter**.

> **Storage Note:** By default, uploaded files are saved to `app/uploads` and extracted text is stored in a local `rag_app.db` SQLite database.

---

## ⚙️ Full Configuration (`.env`)

To enable PostgreSQL, Pinecone vector search, and Groq LLM responses, create a `.env` file in the root directory:

```env
# Database Configuration
# Leave blank to default to local SQLite (sqlite:///./rag_app.db)
DATABASE_URL=postgresql://user:password@localhost:5432/your_database_name

# Pinecone Configuration (Optional - falls back to local search if omitted)
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# Groq API Configuration (Optional - falls back to local matching if omitted)
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
GROQ_PROXY=

# File Storage
UPLOAD_DIR=app/uploads
```

### Setting Up PostgreSQL & Pinecone (If Enabled)

1. **PostgreSQL Database**:
   ```sql
   CREATE DATABASE your_database_name;
   ```
   Run migrations or initialize tables:
   ```bash
   # Using Alembic (recommended)
   alembic upgrade head

   # Or initialize directly via Python
   python -c "from app.db import engine; from app.models.models import Base; Base.metadata.create_all(bind=engine)"
   ```

2. **Pinecone Index**:
   - Create an index in your Pinecone console with:
     - **Dimensions**: `384`
     - **Metric**: `cosine`

---

## 🐳 Docker Production Deployment

### 1. Configure Production Environment
```bash
cp .env.production.template .env.production
nano .env.production  # Add your production API keys and DB credentials
```

### 2. Setup SSL Certificates
Place your `cert.pem` and `key.pem` inside an `ssl/` directory, or generate self-signed certificates for testing:
```bash
mkdir -p ssl
openssl req -x509 -newkey rsa:4096 -keyout ssl/key.pem -out ssl/cert.pem -days 365 -nodes
```

### 3. Deploy
```bash
chmod +x deploy.sh
./deploy.sh
```

#### Useful Docker Commands
```bash
# Start services in background
docker-compose -f docker-compose.prod.yml up -d

# View live logs
docker-compose -f docker-compose.prod.yml logs -f

# Restart or stop services
docker-compose -f docker-compose.prod.yml restart
docker-compose -f docker-compose.prod.yml down
```

- **HTTPS**: `https://localhost`
- **HTTP**: `http://localhost` *(redirects to HTTPS)*
- **Direct API**: `http://localhost:8000`

---

## 📡 API Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Check application health status |
| `GET` | `/test-pinecone/` | Verify Pinecone index connectivity |
| `POST` | `/ask-question/` | Submit a question against uploaded documents |

### Example Request (`POST /ask-question/`)
```json
{
  "question": "What are the key takeaways from the uploaded report?"
}
```

---

## 📁 Project Structure

```text
├── app/
│   ├── models/
│   │   └── models.py       # SQLAlchemy database models
│   ├── uploads/            # Default directory for uploaded documents
│   └── db.py               # Database connection & session setup
├── static/                 # Static assets (CSS, JS, logos, favicons)
│   ├── bolt-logo.png
│   └── bolt_fav.png
├── templates/
│   └── index.html          # Frontend chat & upload interface
├── main.py                 # FastAPI entry point & API routes
├── requirements.txt        # Python dependencies
├── docker-compose.prod.yml # Production Docker orchestration
├── deploy.sh               # Deployment automation script
└── README.md               # Project documentation
```