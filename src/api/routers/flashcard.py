# -*- coding: utf-8 -*-
"""
Flashcard Router
================

Endpoints for generating flashcards from knowledge bases and documents.
Optimized for speed with background processing and context limiting.
"""

import json
import asyncio
import tempfile
import os
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form, BackgroundTasks
from pydantic import BaseModel, Field

from src.logging import get_logger
from src.services.rag.service import RAGService
from src.services.llm.factory import complete
from src.services.llm.config import get_llm_config
from src.services.document_parser import get_document_parser

logger = get_logger("FlashcardAPI")

router = APIRouter(tags=["flashcard"])


class Flashcard(BaseModel):
    """Flashcard model."""
    front: str = Field(..., description="Front of the card (Question/Term)")
    back: str = Field(..., description="Back of the card (Answer/Definition)")


class GenerateRequest(BaseModel):
    """Request model for generation from Knowledge Base."""
    kb_name: str = Field(..., description="Target Knowledge Base Name")
    topic: Optional[str] = Field("General Review", description="Topic to focus on")
    count: int = Field(5, ge=1, le=20, description="Number of cards to generate")
    provider: Optional[str] = Field("gemini", description="LLM Provider to use")


class GenerateResponse(BaseModel):
    """Response model."""
    cards: List[Flashcard]
    source_context: Optional[str] = None


class GenerateFromDocRequest(BaseModel):
    """Request model for generating flashcards from uploaded document content."""
    topic: Optional[str] = Field("General Review", description="Topic/focus area")
    count: int = Field(5, ge=1, le=20, description="Number of cards to generate")
    provider: Optional[str] = Field("gemini", description="LLM Provider to use")


# ============================================
# Generate from Knowledge Base (Optimized)
# ============================================

@router.post("/generate", response_model=GenerateResponse)
async def generate_flashcards(request: GenerateRequest):
    """
    Generate flashcards from a Knowledge Base using LLM (Gemini by default).
    Optimized with reduced context for faster processing.
    """
    logger.info(f"Generating flashcards for KB '{request.kb_name}', Topic: {request.topic}")
    
    context = ""
    try:
        # 1. Retrieve Context (optimized - reduced limit for speed)
        rag_service = RAGService()
        
        search_result = await rag_service.search(
            query=request.topic or "General summary",
            kb_name=request.kb_name,
            limit=3  # Reduced from 5 for speed
        )
        
        context = search_result.get("content", "") or search_result.get("answer", "")
        logger.info(f"Retrieved context length: {len(context)}")
        
        if not context:
            logger.warning("No context found for flashcards")
            raise HTTPException(status_code=404, detail=f"No content found in Knowledge Base '{request.kb_name}' for topic '{request.topic}'. Please ensure the KB is not empty.")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"RAG Retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge Base retrieval failed: {str(e)}")

    # 2. Generate with LLM (optimized prompt for faster response)
    return await _generate_cards_from_context(
        context=context,
        topic=request.topic or "General Review",
        count=request.count,
        provider=request.provider or "gemini"
    )


# ============================================
# Generate from Uploaded Document (Fast Mode)
# ============================================

@router.post("/generate/from-document", response_model=GenerateResponse)
async def generate_flashcards_from_document(
    file: UploadFile = File(...),
    topic: str = Form("General Review"),
    count: int = Form(5),
    provider: str = Form("gemini"),
):
    """
    Generate flashcards directly from an uploaded document (DOCX, PPTX, PDF).
    Optimized for speed with limited context extraction.
    
    Supported file formats:
    - DOCX (Word documents)
    - PPTX (PowerPoint presentations)
    - PDF documents
    """
    logger.info(f"Generating flashcards from uploaded document: {file.filename}")
    
    # Validate file extension
    allowed_extensions = {".docx", ".pptx", ".pdf", ".doc", ".ppt"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: DOCX, PPTX, PDF"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse document (optimized - limited to 5000 chars for speed)
        parser = get_document_parser()
        context = await parser.parse_async(content, file.filename or "document", 5000)
        
        logger.info(f"Extracted {len(context)} characters from document")
        
        if not context or len(context.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Document appears to be empty or contains insufficient text content."
            )
        
        # Generate flashcards from document content
        return await _generate_cards_from_context(
            context=context,
            topic=topic,
            count=count,
            provider=provider
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


async def _generate_cards_from_context(
    context: str,
    topic: str,
    count: int,
    provider: str
) -> GenerateResponse:
    """Generate flashcards from extracted context using LLM (optimized)."""
    
    # Optimized prompt for faster JSON generation
    prompt = f"""Create {count} flashcards about {topic}.

CONTENT:
{context[:2000]}

Output ONLY a JSON array:
[
  {{"front": "Question", "back": "Answer"}}
]

Brief questions, concise answers."""
    
    try:
        llm_response = await complete(
            prompt=prompt,
            system_prompt="You are a flashcard generator. Output ONLY valid JSON arrays, no other text.",
            binding=provider,
            temperature=0.3,
            max_tokens=800,  # Limit tokens for faster generation (~100 per card + buffer)
            max_retries=3,
            retry_delay=0.5  # Faster retry for speed
        )
        
        # Robust JSON extraction
        cleaned_response = llm_response.strip()
        
        # Remove markdown code blocks
        cleaned_response = cleaned_response.replace("```json", "").replace("```", "").strip()
        
        # Find JSON array - look for first [ and last ]
        start_idx = cleaned_response.find('[')
        end_idx = cleaned_response.rfind(']')
        
        if start_idx == -1 or end_idx == -1:
            raise ValueError("No JSON array found in response")
        
        json_str = cleaned_response[start_idx:end_idx + 1]
        
        # Parse JSON
        cards_data = json.loads(json_str)
        
        # Validate structure
        cards = []
        for item in cards_data:
            if "front" in item and "back" in item:
                cards.append(Flashcard(front=item["front"], back=item["back"]))
                
        if not cards:
             raise ValueError("No valid flashcards parsed from LLM response")
                
        return GenerateResponse(cards=cards, source_context=context[:200] + "...")
        
    except json.JSONDecodeError as e:
        logger.error(f"LLM JSON Parse Error. Response: {llm_response[:500]}")
        raise HTTPException(status_code=500, detail="Generated content was not valid JSON.")
    except Exception as e:
        logger.error(f"Generation failed: {e}")

        raise HTTPException(status_code=500, detail=f"Flashcard generation failed: {str(e)}")


# ============================================
# Background Generation (For very large documents)
# ============================================

class BackgroundGenerationRequest(BaseModel):
    """Request model for background generation."""
    file_id: str  # Uploaded file reference
    topic: str = "General Review"
    count: int = 5


@router.post("/generate/background")
async def generate_flashcards_background(
    request: BackgroundGenerationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate flashcards in background (for large documents).
    Returns a task_id for polling status.
    """
    # This would require file storage implementation
    # For now, return a placeholder response
    return {
        "status": "pending",
        "message": "Background generation feature coming soon",
        "task_id": None
    }


# ============================================
# Parse Document Text (Fast Preview)
# ============================================

class ParseDocumentResponse(BaseModel):
    """Response model for document parsing."""
    filename: str
    text_preview: str = Field(..., description="First 500 characters of extracted text")
    text_length: int
    success: bool


@router.post("/parse-document", response_model=ParseDocumentResponse)
async def parse_uploaded_document(
    file: UploadFile = File(...),
    max_chars: int = Form(500)  # Just a preview, not full extraction
):
    """
    Parse an uploaded document and return extracted text (fast preview).
    """
    logger.info(f"Parsing uploaded document preview: {file.filename}")
    
    # Validate file extension
    allowed_extensions = {".docx", ".pptx", ".pdf", ".doc", ".ppt"}
    file_ext = os.path.splitext(file.filename or "")[1].lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: DOCX, PPTX, PDF"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse document (fast preview mode)
        parser = get_document_parser()
        text = await parser.parse_async(content, file.filename or "document", max_chars)
        
        preview = text[:500] + ("..." if len(text) >= 500 else "")
        
        return ParseDocumentResponse(
            filename=file.filename or "document",
            text_preview=preview,
            text_length=len(text),
            success=True
        )
        
    except Exception as e:
        logger.error(f"Document parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {str(e)}")


# ============================================
# Persistence Endpoints
# ============================================

from src.services.flashcard.manager import FlashcardManager, FlashcardSet

# Dependency
def get_manager():
    return FlashcardManager()


@router.get("/sets", response_model=List[FlashcardSet])
async def list_flashcard_sets(manager: FlashcardManager = Depends(get_manager)):
    """List all flashcard sets."""
    return manager.list_sets()


@router.get("/sets/{set_id}", response_model=FlashcardSet)
async def get_flashcard_set(set_id: str, manager: FlashcardManager = Depends(get_manager)):
    """Get a specific flashcard set by ID."""
    fs = manager.get_set(set_id)
    if not fs:
        raise HTTPException(status_code=404, detail="Flashcard set not found")
    return fs


class CreateSetRequest(BaseModel):
    kb_name: str
    topic: str
    cards: List[dict] # {front: str, back: str}


@router.post("/sets", response_model=FlashcardSet)
async def create_flashcard_set(request: CreateSetRequest, manager: FlashcardManager = Depends(get_manager)):
    """Save a new flashcard set."""
    return manager.create_set(
        kb_name=request.kb_name,
        topic=request.topic,
        cards_data=request.cards
    )


@router.delete("/sets/{set_id}")
async def delete_flashcard_set(set_id: str, manager: FlashcardManager = Depends(get_manager)):
    """Delete a flashcard set."""
    success = manager.delete_set(set_id)
    if not success:
        raise HTTPException(status_code=404, detail="Flashcard set not found")
    return {"message": "Flashcard set deleted successfully"}


@router.post("/sets/{set_id}/cards/{card_id}/review", response_model=FlashcardSet)
async def review_flashcard(set_id: str, card_id: str, manager: FlashcardManager = Depends(get_manager)):
    """Mark a card as reviewed (increment count, update timestamp)."""
    updated_set = manager.update_card_review(set_id, card_id)
    if not updated_set:
        raise HTTPException(status_code=404, detail="Flashcard set or card not found")
    return updated_set


@router.put("/sets/{set_id}/cards/{card_id}/star", response_model=FlashcardSet)
async def toggle_star_flashcard(set_id: str, card_id: str, manager: FlashcardManager = Depends(get_manager)):
    """Toggle star status of a card."""
    updated_set = manager.toggle_card_star(set_id, card_id)
    if not updated_set:
        raise HTTPException(status_code=404, detail="Flashcard set or card not found")
    return updated_set

