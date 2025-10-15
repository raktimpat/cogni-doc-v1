from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import NotFound
from google.cloud import discoveryengine_v1 as discoveryengine
from google.cloud import storage
from fastapi import HTTPException
import os
from dotenv import load_dotenv
from vertexai.generative_models import GenerativeModel
import uuid
load_dotenv()


# --------------- Load Environment Variables -----------------#

GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
DOC_AI_LOCATION = os.environ.get("DOC_AI_LOCATION", "us") # e.g., 'us' or 'eu'
DOC_AI_PROCESSOR_ID = os.environ.get("DOC_AI_PROCESSOR_ID", "")
ENGINE_ID = os.environ.get("VERTEX_AI_RAG_APP_ID")
VERTEX_AI_APP_LOCATION = os.environ.get("VERTEX_AI_APP_LOCATION")
DATASTORE_BUCKET = os.environ.get("DATASTORE_BUCKET", "")



def docai_parser(file_content: bytes, mime_type: str, parser_type: str) -> str | dict:
    docai_client = documentai.DocumentProcessorServiceClient()
    DOCAI_PROCESSOR_PATH = docai_client.processor_path(GCP_PROJECT_ID, DOC_AI_LOCATION, DOC_AI_PROCESSOR_ID)

    raw_document = documentai.RawDocument(content=file_content, mime_type=mime_type)
    request = documentai.ProcessRequest(name=DOCAI_PROCESSOR_PATH, raw_document=raw_document)
    
    try:
        result = docai_client.process_document(request=request)
        document = result.document
        
        if parser_type == "invoice":
            extracted_data = {"entities": []}
            for entity in document.entities:
                extracted_data["entities"].append({
                    "type": entity.type_,
                    "value": entity.mention_text,
                    "confidence": round(entity.confidence, 2)
                })
        elif parser_type == "document":
            extracted_data = document.text

        return extracted_data
    except Exception as e:
        print(f"Error processing with Document AI: {e}")
        raise HTTPException(status_code=500, detail=f"Document AI processing failed: {e}")


class VertexApplication:

    def __init__(self):
        discovery_engine_client_options = (
            ClientOptions(api_endpoint=f"{VERTEX_AI_APP_LOCATION}-discoveryengine.googleapis.com")
            if VERTEX_AI_APP_LOCATION != "global"
            else None
        )
        self.discovery_engine_client = discoveryengine.ConversationalSearchServiceClient(client_options=discovery_engine_client_options)
        self.serving_config = f"projects/{GCP_PROJECT_ID}/locations/{VERTEX_AI_APP_LOCATION}/collections/default_collection/engines/{ENGINE_ID}/servingConfigs/default_config"



    def document_summarizer(self, query_text: str, document_text:str) -> dict:
        try:
            # Initialize the generative model
            model = GenerativeModel("gemini-2.0-flash-lite-001")
            
            prompt = f"""
            Based *only* on the following document text, answer the user's question.
            Do not use any external knowledge. If the answer cannot be found in the text,
            state that the information is not present in the document.

            --- DOCUMENT TEXT START ---
            {document_text}
            --- DOCUMENT TEXT END ---

            --- QUESTION ---
            {query_text}
            """

            response = model.generate_content(prompt)
            return {"answer": response.text}
        except Exception as e:
            print(f"Error querying on single document with Vertex AI: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to query on single document: {e}")




    def store_summarizer(self, query_text: str) -> dict:
        query_understanding_spec = discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec(
        query_rephraser_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryRephraserSpec(
            disable=False,  
            max_rephrase_steps=1,  
        ),
        # Optional: Classify query types
        query_classification_spec=discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec(
            types=[
                discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.ADVERSARIAL_QUERY,
                discoveryengine.AnswerQueryRequest.QueryUnderstandingSpec.QueryClassificationSpec.Type.NON_ANSWER_SEEKING_QUERY,
            ]  
        ),
    )

        answer_generation_spec = discoveryengine.AnswerQueryRequest.AnswerGenerationSpec(
        ignore_adversarial_query=False,  
        ignore_non_answer_seeking_query=False, 
        ignore_low_relevant_content=False, 
        model_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.ModelSpec(
            model_version="gemini-2.0-flash-001/answer_gen/v1", 
        ),
        prompt_spec=discoveryengine.AnswerQueryRequest.AnswerGenerationSpec.PromptSpec(
            preamble="Give a detailed answer.",  
        ),
        include_citations=True,  
        answer_language_code="en",  
    )

        # Initialize request argument(s)
        request = discoveryengine.AnswerQueryRequest(
        serving_config=self.serving_config,
        query=discoveryengine.Query(text=query_text),
        session=None,  # Optional: include previous session ID to continue a conversation
        query_understanding_spec=query_understanding_spec,
        answer_generation_spec=answer_generation_spec,
        user_pseudo_id="user-pseudo-id",  # Optional: Add user pseudo-identifier for queries.
    )

        # Make the request
        try:
            response = self.discovery_engine_client.answer_query(request)
            print(response)
            if not response:
                return {"answer": "Could not generate an answer for the given query from the available documents."}

            return {"answer": response.answer.answer_text}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to query on store: {e}")
        
        return response
    

    def fine_tune_uploader(self, contents: bytes, content_type: str,  filename: str) -> dict:

        if not DATASTORE_BUCKET or DATASTORE_BUCKET.startswith("REPLACE"):
            raise HTTPException(
                status_code=500,
                detail="GCS bucket for fine-tuning is not configured on the server."
            )

        try:
            
            # Step 1: Upload the file to Google Cloud Storage
            storage_client = storage.Client(project=GCP_PROJECT_ID)
            bucket = storage_client.bucket(DATASTORE_BUCKET)
            blob_name = f"upload/{uuid.uuid4()}-{filename}"
            blob = bucket.blob(blob_name)
            
            
            blob.upload_from_string(contents, content_type=content_type)
            gcs_uri = f"gs://{DATASTORE_BUCKET}/{blob_name}"
            print(f"File '{filename}' uploaded to '{gcs_uri}'.")

            return "File uploaded successfully"
        except NotFound:
            raise HTTPException(
                status_code=404,
                detail=f"GCS bucket '{DATASTORE_BUCKET}' not found. Please create it and grant permissions."
            )