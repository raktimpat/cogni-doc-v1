import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from utils import docai_parser, VertexApplication

app = FastAPI(
    title="CogniDoc API",
    description="API for parsing invoices and analyzing research papers.",
    version="1.0.0"
)

# CORS (Cross-Origin Resource Sharing) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

vertexai = VertexApplication()

# --- API Endpoints ---
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "CogniDoc API is running!"}


@app.post("/parse-invoice/", summary="Parse an Invoice Document")
async def parse_invoice(file: UploadFile = File(...)):
    if not (file.content_type.startswith('image/') or file.content_type == 'application/pdf'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image or a PDF.")
    
    try:
        parser_type="invoice"
        contents = await file.read()
        parsed_data = docai_parser(contents, file.content_type, parser_type)
        return parsed_data
    except Exception as e:
        raise HTTPException(status_code=500, detail="Parser not available")


@app.post("/analyze-paper/", summary="Analyze a Research Paper")
async def analyze_paper(
    prompt_type: str = Form(...),
    custom_question: str = Form(None),
    file: UploadFile = File(None) 
):
    try:
        if prompt_type == 'summary':
            query = "Generate a concise, one-paragraph summary of this research paper."
        elif prompt_type == 'questions':
            query = "Generate 5 insightful study questions based on this paper, each with a detailed answer."
        elif prompt_type == 'custom':
            if not custom_question:
                raise HTTPException(status_code=400, detail="A custom question is required.")
            query = custom_question
        else:
            raise HTTPException(status_code=400, detail="Invalid prompt type specified.")
        
        if file and file.filename:
            if not (file.content_type.startswith('image/') or file.content_type == 'application/pdf' or file.content_type == 'text/plain'):
                raise HTTPException(status_code=400, detail="Invalid file type for analysis. Please upload an image, PDF, or text file.")
            
            contents = await file.read()
            parser_type = "document"
            document_text = docai_parser(contents, file.content_type, parser_type)
            response = vertexai.document_summarizer(query, document_text)
            return response
        else:
            rag_response = vertexai.store_summarizer(query)
            return rag_response
    except:
        raise HTTPException(status_code=500, detail="Assistant not avaiable")


@app.post("/datastore_qa/", summary="General purpose chat")
async def chat(query: str = Form(...)):
    """Handles general chat questions with a generative model."""
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        rag_response = vertexai.store_summarizer(query)
        return rag_response
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get chat response: {e}")



@app.post("/upload-finetune/", summary="Upload File to Datastore")
async def upload_finetune_file(file: UploadFile = File(...)):
    """
    Uploads a file to a GCS bucket and triggers an import job
    to add the document to the Vertex AI Search datastore.
    """
    try:
        contents = await file.read()
        response = vertexai.fine_tune_uploader(contents, file.content_type, file.filename)
        return {
            "message": f"'{file.filename}' {response}."
        }
    except Exception as e:
        print(f"Error during fine-tuning upload: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file for fine-tuning: {e}")


