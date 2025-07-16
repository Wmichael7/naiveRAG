from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG components
from rag_qa import qa

app = FastAPI(
    title="Naive RAG API",
    description="A simple RAG implementation with OpenAI and FAISS",
    version="1.0.0",
)


class QuestionRequest(BaseModel):
    question: str


class QuestionResponse(BaseModel):
    answer: str
    sources: List[str]


@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and AWS"""
    return {"status": "healthy", "service": "naive-rag"}


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get RAG-powered answer with sources"""
    try:
        result = qa.invoke(request.question)

        # Extract sources
        sources = []
        for doc in result["source_documents"]:
            source = doc.metadata.get("source", "Unknown")
            sources.append(source)

        return QuestionResponse(answer=result["result"], sources=sources)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing question: {str(e)}"
        )


@app.get("/")
async def root():
    """Root endpoint with basic info"""
    return {
        "message": "Naive RAG API",
        "version": "1.0.0",
        "endpoints": {"health": "/health", "ask": "/ask"},
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
