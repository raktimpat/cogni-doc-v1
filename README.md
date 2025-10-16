# CogniDoc

A FastAPI backend for parsing and querying documents using Google Cloud AI services.

## Features

- **Invoice Parsing**: Extract structured data from invoice images/PDFs using Document AI
- **Research Q&A (RAG)**: Query a pre-indexed knowledge base with Vertex AI Search using Retrieval-Augmented Generation
- **Document Q&A**: Upload a document and ask questions about its content using Gemini
- **Knowledge Base Management**: Add documents to your Vertex AI Search datastore

## How RAG Works

This service uses Retrieval-Augmented Generation (RAG) to answer questions based on your documents:

1. **Indexing**: Documents are uploaded to Google Cloud Storage and indexed in Vertex AI Search
2. **Retrieval**: When you ask a question, the system searches the indexed documents for relevant passages
3. **Generation**: Retrieved context is passed to Gemini AI to generate accurate, grounded answers
4. **Response**: You get answers based on your actual documents, not just the AI's training data

This approach ensures answers are factual and traceable to specific source documents.

## Prerequisites

- Python 3.11+
- Google Cloud CLI (`gcloud`)
- GCP project with enabled APIs:
  - Document AI
  - Vertex AI Search & Conversation
  - Cloud Storage

## Setup

1. **Clone and install**

```bash
git clone https://github.com/your-repo/cognidoc.git
cd cognidoc
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

2. **Configure environment**

Create `backend/.env`:

```bash
GCP_PROJECT_ID="your-project-id"
DOC_AI_LOCATION="us"
DOC_AI_PROCESSOR_ID="your-processor-id"
VERTEX_AI_LOCATION="us-central1"
VERTEX_AI_DATA_STORE_ID="your-datastore-id"
GCS_FINETUNE_BUCKET_NAME="your-bucket-name"
```

3. **Run locally**

```bash
cd backend
uvicorn app.main:app --reload
```

API available at `http://127.0.0.1:8000`

## API Endpoints

- `POST /parse/invoice/` - Parse invoice documents
- `POST /rag/research-assistant/` - Ask questions with optional document upload
- `POST /rag/datastore-qa/` - Query the indexed knowledge base
- `POST /upload/finetune/` - Upload documents to the datastore

## Deployment

Deploy to Google Cloud Run:

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cognidoc
gcloud run deploy cognidoc --image gcr.io/YOUR_PROJECT_ID/cognidoc --platform managed
```
