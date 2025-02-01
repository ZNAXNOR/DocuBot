from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import os
from document_loader import load_and_split_document
from qa_system import QASystem

app = FastAPI()

# Global variable to store document chunks in memory
document_chunks = []

# System Prompt to Improve AI Responses
SYSTEM_PROMPT = """
You are an AI document assistant. Answer questions based on uploaded documents.
Ensure responses are at least 100 words long and provide well-structured, detailed answers.
Follow these special instructions:
- If the user says "Summarize", provide a concise summary in bullet points.
- If the user says "Explain like I'm five (ELI5)", simplify your explanation.
- If the user says "Give examples", always include at least two real-world examples.
- If the document lacks enough information, clearly state so instead of guessing.
"""


# Request model for the ask endpoint
class QuestionRequest(BaseModel):
    question: str


@app.get("/")
async def root():
    """
    Root endpoint for the app.
    """
    return {"message": "Welcome to the AI-Powered Q&A System!"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a document and process it into chunks.
    """
    global document_chunks

    # Save the uploaded file temporarily
    try:
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(file.file.read())

        # Load and split the document into chunks
        document_chunks = load_and_split_document(temp_file_path)

        # Clean up the temporary file
        os.remove(temp_file_path)

        if not document_chunks:
            raise HTTPException(status_code=400, detail="Failed to process document.")

        return {"message": "Document uploaded and processed successfully!", "chunks": len(document_chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/ask/")
async def ask_question(request: QuestionRequest):
    """
    Endpoint to answer a question based on uploaded document context.
    """
    global document_chunks

    if not document_chunks:
        raise HTTPException(
            status_code=400,
            detail="No document uploaded. Please upload a document first."
        )

    # Extract the question from the request
    question = request.question
    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    # Concatenate chunks to create context
    context = " ".join(chunk.page_content for chunk in document_chunks)

    # Initialize & Combine system prompt with user query the Q&A system and get the answer
    qa_system = QASystem()
    full_question = f"{SYSTEM_PROMPT}\n\nUser: {question}\n\nAI:"
    answer = qa_system.answer_question(full_question, context)

    # Ensure minimum word count
    def ensure_min_length(response, min_words=100):
        words = response.split()
        if len(words) < min_words:
            return f"{response}\n\nPlease elaborate further to ensure a more complete answer."
        return response

    return {"question": question, "answer": ensure_min_length(answer)}
