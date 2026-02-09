"""
Question Generation API Router
Handles custom question generation, exam mimicry, and document-based quiz generation
"""

import json
import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from src.agents.question.coordinator import AgentCoordinator
from src.tools.question.exam_mimic import mimic_exam_questions
from src.logging import get_logger
from src.services.llm import get_llm_config
from src.services.llm.factory import complete
from src.services.config import load_config_with_main
from src.services.document_parser import get_document_parser
from src.services.rag.service import RAGService

router = APIRouter()

# Initialize logger
project_root = Path(__file__).parent.parent.parent.parent
config = load_config_with_main("main.yaml", project_root)
log_dir = config.get("paths", {}).get("user_log_dir") or config.get("logging", {}).get("log_dir")
logger = get_logger("QuestionAPI", level="INFO", log_dir=log_dir)

# KB base directory
_kb_base_dir = project_root / "data" / "knowledge_bases"


async def get_kb_content(kb_name: str, max_chars: int = 6000) -> str:
    """
    Get content directly from Knowledge Base's content_list.
    Falls back to RAG search if content_list is empty.
    Falls back to parsing raw documents if RAG is also empty.
    
    Args:
        kb_name: Knowledge base name
        max_chars: Maximum characters to return (for speed)
        
    Returns:
        Combined text content from all documents in the KB
    """
    content_list_path = _kb_base_dir / kb_name / "content_list"
    
    # Try content_list first
    if content_list_path.exists() and list(content_list_path.glob("*.json")):
        all_content = []
        
        # Read all JSON files in content_list
        try:
            for json_file in content_list_path.glob("*.json"):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # Extract content from different possible structures
                    if isinstance(data, dict):
                        # Try common content fields
                        content = (data.get('content') or 
                                 data.get('text') or 
                                 data.get('full_text') or 
                                 data.get('markdown') or
                                 data.get('data', {}).get('content') or
                                 str(data))
                        all_content.append(content)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                content = (item.get('content') or 
                                         item.get('text') or 
                                         item.get('markdown') or
                                         str(item))
                                all_content.append(content)
                except Exception as e:
                    logger.warning(f"Failed to read {json_file}: {e}")
                    continue
            
            # Combine all content
            combined = "\n\n".join(all_content)
            
            # Truncate for speed if needed
            if len(combined) > max_chars:
                logger.info(f"Truncating KB content from {len(combined)} to {max_chars} chars")
                combined = combined[:max_chars]
            
            if combined and len(combined.strip()) > 50:
                return combined
                
        except Exception as e:
            logger.error(f"Failed to read KB content from content_list: {e}")
    
    # Fallback 1: Try RAG search
    logger.info(f"content_list empty or failed, trying RAG search for KB '{kb_name}'")
    rag_content = await get_kb_content_via_rag(kb_name, max_chars)
    if rag_content and len(rag_content.strip()) > 50:
        return rag_content
    
    # Fallback 2: Parse raw documents directly
    logger.info(f"RAG search returned empty, trying to parse raw documents for KB '{kb_name}'")
    raw_dir = _kb_base_dir / kb_name / "raw"
    if raw_dir.exists():
        all_content = []
        for file_path in raw_dir.iterdir():
            if file_path.is_file():
                try:
                    logger.info(f"Parsing raw document: {file_path.name}")
                    parser = get_document_parser()
                    content = await parser.parse_async(
                        file_path.read_bytes(),
                        file_path.name,
                        max_chars=max_chars
                    )
                    if content and len(content.strip()) > 50:
                        all_content.append(content)
                        # Stop if we have enough content
                        if len("\n\n".join(all_content)) >= max_chars:
                            break
                except Exception as e:
                    logger.warning(f"Failed to parse {file_path.name}: {e}")
                    continue
        
        combined = "\n\n".join(all_content)
        if len(combined) > max_chars:
            combined = combined[:max_chars]
        
        if combined and len(combined.strip()) > 50:
            return combined
    
    # All fallbacks failed
    return ""


async def get_kb_content_via_rag(kb_name: str, max_chars: int = 6000) -> str:
    """
    Fallback: Get content from KB via RAG search (optimized for speed).
    Uses a single comprehensive query for faster retrieval.
    """
    rag_service = RAGService()
    
    try:
        # Single optimized query for speed
        search_result = await rag_service.search(
            query="Provide a comprehensive overview of all main topics, concepts, and key information",
            kb_name=kb_name,
            limit=5,  # Limit chunks for faster retrieval
            mode="hybrid"
        )
        content = search_result.get("content", "") or search_result.get("answer", "")
        
        if len(content) > max_chars:
            content = content[:max_chars]
        
        return content
    except Exception as e:
        logger.warning(f"RAG search failed: {e}")
        return ""


class GenerateRequest(BaseModel):
    kb_name: str
    difficulty: str = "medium"
    count: int = 5


class QuestionResponse(BaseModel):
    question: str
    answer: Optional[str] = None
    explanation: Optional[str] = None


class GenerateResponse(BaseModel):
    questions: list


# ============================================
# Generate from Knowledge Base (Exam Style)
# ============================================

@router.post("/generate/from-kb", response_model=GenerateResponse)
async def generate_exam_from_kb(request: GenerateRequest):
    """
    Generate exam-style questions from a Knowledge Base.
    Creates short-answer/essay questions, not multiple choice.
    """
    logger.info(f"Generating exam questions from KB: {request.kb_name}, difficulty: {request.difficulty}")
    
    try:
        # Get context directly from KB content_list (optimized limit)
        context = await get_kb_content(request.kb_name, max_chars=6000)
        logger.info(f"Retrieved context length: {len(context)}")
        
        if not context or len(context.strip()) < 50:
            raise HTTPException(
                status_code=404,
                detail=f"No content found in Knowledge Base '{request.kb_name}'. Please ensure the KB is not empty."
            )
        
        return await _generate_exam_questions(
            context=context,
            difficulty=request.difficulty,
            count=request.count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


# ============================================
# Generate from Uploaded Document (Exam Style)
# ============================================

@router.post("/generate/from-document", response_model=GenerateResponse)
async def generate_exam_from_document(
    file: UploadFile = File(...),
    difficulty: str = Form("medium"),
    count: int = Form(5),
):
    """
    Generate multiple-choice questions (MCQs) directly from an uploaded document (DOCX, PPTX, PDF).
    """
    logger.info(f"Generating MCQ questions from document: {file.filename}")
    
    # Validate file extension
    allowed_extensions = {".docx", ".pptx", ".pdf", ".doc", ".ppt"}
    file_ext = Path(file.filename or "").suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported formats: DOCX, PPTX, PDF"
        )
    
    try:
        # Read file content
        content = await file.read()
        
        # Parse document (limited to 5000 chars for speed)
        parser = get_document_parser()
        context = await parser.parse_async(content, file.filename or "document", 5000)
        
        logger.info(f"Extracted {len(context)} characters from document")
        
        if not context or len(context.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Document appears to be empty or contains insufficient text content."
            )
        
        return await _generate_mcq_from_document_context(
            context=context,
            difficulty=difficulty,
            count=count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")


async def _generate_mcq_from_document_context(
    context: str,
    difficulty: str,
    count: int
) -> GenerateResponse:
    """Generate MCQ questions from document context."""
    
    diff_instructions = {
        "easy": "Focus on basic recall and recognition",
        "medium": "Focus on understanding and comprehension",
        "hard": "Focus on analysis, synthesis, and application"
    }
    
    diff_guide = diff_instructions.get(difficulty.lower(), diff_instructions["medium"])
    
    # Concise difficulty hints for speed
    diff_hints = {
        "easy": "basic facts/definitions",
        "medium": "understanding/application", 
        "hard": "analysis/critical thinking"
    }
    
    hint = diff_hints.get(difficulty.lower(), "understanding/application")
    
    prompt = f"""Generate {count} MCQs ({difficulty.upper()}: {hint}).

CONTENT:
{context[:1500]}

JSON:
[{{"question":"?","options":["A","B","C","D"],"correct_answer":0,"explanation":"?"}}]

{difficulty.upper()} difficulty. 4 plausible options. Valid JSON only."""
    
    try:
        # Get LLM configuration - use faster 8B model for quiz generation to avoid rate limits
        llm_config = get_llm_config()
        # Use llama-3.1-8b-instant for speed and to avoid rate limits
        binding = "groq"  # Keep groq provider
        model_override = "llama-3.1-8b-instant"  # Use smaller, faster model
        
        # Generate questions using LLM (optimized for speed)
        llm_response = await complete(
            prompt=prompt,
            system_prompt=f"Expert quiz generator. {difficulty.upper()} difficulty. JSON only.",
            binding=binding,
            model=model_override,  # Override with faster model
            temperature=0.3,  # Lower for faster, more focused responses
            max_tokens=min(count * 120 + 150, 1500),  # Slightly reduced
            max_retries=2,  # Reduced retries for speed
            retry_delay=0.3  # Even faster retry
        )
        
        cleaned = llm_response.strip().replace("```json", "").replace("```", "").strip()
        
        # Extract JSON list if embedded in text
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        
        if start == -1 or end == -1:
             # Try parsing as a single object if array not found
            if cleaned.strip().startswith('{') and cleaned.strip().endswith('}'):
                 cleaned = f"[{cleaned}]"
                 start = 0
                 end = len(cleaned) - 1
            else:
                logger.error(f"No JSON array found in response: {llm_response[:200]}")
                raise ValueError("Could not find JSON array in LLM response")
        
        json_str = cleaned[start:end + 1]
        questions_data = json.loads(json_str)
        
        if not isinstance(questions_data, list):
            raise ValueError("Response is not a list")
            
        logger.info(f"Successfully generated {len(questions_data)} MCQ questions")
        return OldGenerateResponse(questions=questions_data)
        
    except Exception as e:
        logger.error(f"MCQ generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


async def _generate_exam_questions(
    context: str,
    difficulty: str,
    count: int
) -> GenerateResponse:
    """Generate exam-style questions (short answer/essay, not MCQ)."""
    
    # Difficulty-specific instructions
    difficulty_settings = {
        "easy": {
            "question_type": "short answer (1-2 sentences)",
            "complexity": "basic understanding and recall",
            "keywords": "define, describe, explain, identify"
        },
        "medium": {
            "question_type": "medium-length answer (2-4 sentences)",
            "complexity": "understanding and application of concepts",
            "keywords": "compare, contrast, analyze, discuss"
        },
        "hard": {
            "question_type": "essay/extended answer (paragraph or more)",
            "complexity": "critical thinking, synthesis, evaluation",
            "keywords": "evaluate, justify, critique, propose"
        }
    }
    
    settings = difficulty_settings.get(difficulty.lower(), difficulty_settings["medium"])
    
    # Ultra-concise prompt for speed
    prompt = f"""Generate {count} {difficulty.upper()} exam questions.

CONTENT:
{context[:1500]}

JSON:
[{{"question":"?","answer":"Sample","explanation":"Hint"}}]

Written answers. Valid JSON only."""
    
    try:
        # Use faster 8B model to avoid rate limits
        binding = "groq"
        model_override = "llama-3.1-8b-instant"
        
        # Generate questions using LLM (optimized for speed)
        llm_response = await complete(
            prompt=prompt,
            system_prompt=f"Exam question generator. {difficulty.upper()}. JSON only.",
            binding=binding,
            model=model_override,
            temperature=0.3,
            max_tokens=min(count * 150 + 150, 1500),
            max_retries=2,
            retry_delay=0.3
        )
        
        logger.info(f"LLM response length: {len(llm_response)}")
        
        # Clean and extract JSON
        cleaned = llm_response.strip()
        cleaned = cleaned.replace("```json", "").replace("```", "").strip()
        
        # Try to find JSON array
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        
        if start == -1 or end == -1:
            # Try alternative: look for first { and last }
            start_obj = cleaned.find('{')
            end_obj = cleaned.rfind('}')
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                # Try parsing as a single object
                logger.warning("Response contains object instead of array, wrapping in array")
                json_str = cleaned[start_obj:end_obj + 1]
                try:
                    obj = json.loads(json_str)
                    questions_data = [obj]
                except json.JSONDecodeError:
                    logger.error(f"No JSON found in response: {llm_response[:200]}")
                    raise ValueError("Could not find JSON in LLM response")
            else:
                logger.error(f"No JSON array found in response: {llm_response[:200]}")
                raise ValueError("Could not find JSON array in LLM response")
        else:
            json_str = cleaned[start:end + 1]
            questions_data = json.loads(json_str)
        
        # Validate
        if not isinstance(questions_data, list):
            raise ValueError("Response is not a list")
        
        if len(questions_data) == 0:
            raise ValueError("No questions generated")
        
        logger.info(f"Successfully generated {len(questions_data)} exam questions")
        return GenerateResponse(questions=questions_data)
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON Parse Error: {e}")
        logger.error(f"Raw response: {llm_response[:500]}")
        raise HTTPException(status_code=500, detail=f"Invalid JSON response from LLM: {str(e)}")
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        if "JSON array" in str(e):
            raise HTTPException(status_code=500, detail=str(e))
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


# ============================================
# Original REST Endpoint (MCQ - kept for backward compatibility)
# ============================================

class QuestionRequirement(BaseModel):
    knowledge_point: str
    difficulty: str = "medium"
    question_type: str = "choice"
    additional_requirements: Optional[str] = None


class OldGenerateRequest(BaseModel):
    requirement: QuestionRequirement
    count: int = 1
    kb_name: Optional[str] = None


class OldQuestionResponse(BaseModel):
    question: str
    options: list
    correct_answer: int
    explanation: Optional[str] = None


class OldGenerateResponse(BaseModel):
    questions: list


@router.post("/generate", response_model=OldGenerateResponse)
async def generate_questions_rest(request: OldGenerateRequest):
    """Legacy endpoint for MCQ generation (for backward compatibility)."""
    logger.info(f"Legacy REST: Generating {request.count} MCQ questions for KB '{request.kb_name}'")
    
    try:
        context = ""
        if request.kb_name:
            try:
                rag_service = RAGService()
                search_result = await rag_service.search(
                    query=request.requirement.knowledge_point or "General knowledge",
                    kb_name=request.kb_name,
                    limit=3
                )
                context = search_result.get("content", "") or search_result.get("answer", "")
            except Exception as e:
                logger.warning(f"Failed to retrieve context: {e}")
        
        return await _generate_mcq_questions(
            context=context,
            requirement=request.requirement,
            count=request.count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"MCQ generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


async def _generate_mcq_questions(
    context: str,
    requirement: QuestionRequirement,
    count: int
) -> OldGenerateResponse:
    """Generate MCQ questions (legacy function)."""
    
    diff_instructions = {
        "easy": "Focus on basic recall and recognition",
        "medium": "Focus on understanding and comprehension",
        "hard": "Focus on analysis, synthesis, and application"
    }
    
    diff_guide = diff_instructions.get(requirement.difficulty.lower(), diff_instructions["medium"])
    
    prompt = f"""Generate {count} MCQs ({requirement.difficulty.upper()}).

Topic: {requirement.knowledge_point}

CONTENT:
{(context if context else requirement.knowledge_point)[:1500]}

JSON: [{{"question":"?","options":["A","B","C","D"],"correct_answer":0,"explanation":"?"}}]

{requirement.difficulty.upper()} difficulty. Valid JSON only."""
    
    try:
        # Use faster 8B model to avoid rate limits
        llm_response = await complete(
            prompt=prompt,
            system_prompt=f"Quiz generator. {requirement.difficulty.upper()}. JSON only.",
            binding="groq",
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=min(count * 120 + 150, 1500),
            max_retries=2,
            retry_delay=0.3
        )
        
        cleaned = llm_response.strip().replace("```json", "").replace("```", "").strip()
        start = cleaned.find('[')
        end = cleaned.rfind(']')
        
        if start == -1 or end == -1:
            raise ValueError("No JSON array found")
        
        json_str = cleaned[start:end + 1]
        questions_data = json.loads(json_str)
        
        if not isinstance(questions_data, list):
            raise ValueError("Response is not a list")
        
        logger.info(f"Generated {len(questions_data)} MCQ questions")
        return OldGenerateResponse(questions=questions_data)
        
    except Exception as e:
        logger.error(f"MCQ generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


# ============================================
# WebSocket Endpoints
# ============================================

@router.websocket("/generate")
async def websocket_generate(websocket: WebSocket):
    """WebSocket endpoint for question generation."""
    await websocket.accept()
    logger.info("Question generation WebSocket accepted")
    
    try:
        data = await websocket.receive_json()
        req_data = data.get("requirement", {})
        count = data.get("count", 1)
        kb_name = data.get("kb_name")
        
        requirement = {
            "knowledge_point": req_data.get("knowledge_point"),
            "difficulty": req_data.get("difficulty", "medium"),
            "question_type": req_data.get("question_type", "choice"),
        }
        
        llm_config = get_llm_config()
        
        async def ws_callback(msg_type: str, data: Any):
            try:
                payload = {"type": msg_type, "timestamp": datetime.now().isoformat()}
                if isinstance(data, dict):
                    payload.update(data)
                else:
                    payload["message"] = str(data)
                await websocket.send_json(payload)
            except Exception as ex:
                logger.error(f"WS callback failed: {ex}")

        coordinator = AgentCoordinator(
            api_key=llm_config.api_key,
            base_url=llm_config.base_url,
            api_version=getattr(llm_config, "api_version", None),
            kb_name=kb_name,
        )
        
        coordinator.set_ws_callback(ws_callback)
        
        results = await coordinator.generate_questions_custom(
            requirement=requirement,
            num_questions=count
        )
        
        if results.get("completed", 0) > 0:
            for q in results.get("questions", []):
                await websocket.send_json({
                    "type": "result",
                    "question_id": q.get("question_id"),
                    "question": q.get("question"),
                })
            await websocket.send_json({"type": "complete", "message": "Generation completed"})
        else:
            await websocket.send_json({"type": "error", "message": "Failed to generate questions"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.websocket("/mimic")
async def websocket_mimic(websocket: WebSocket):
    """WebSocket endpoint for exam mimicry generation."""
    await websocket.accept()
    logger.info("Exam mimicry WebSocket accepted")
    
    try:
        data = await websocket.receive_json()
        mode = data.get("mode")
        kb_name = data.get("kb_name")
        max_questions = data.get("max_questions")
        
        async def ws_callback(msg_type: str, data: Any):
            try:
                payload = {"type": msg_type, "timestamp": datetime.now().isoformat()}
                if isinstance(data, dict):
                    payload.update(data)
                else:
                    payload["message"] = str(data)
                await websocket.send_json(payload)
            except Exception as ex:
                logger.error(f"WS callback failed: {ex}")

        if mode == "upload":
            import base64
            import tempfile
            
            pdf_data = data.get("pdf_data")
            pdf_name = data.get("pdf_name", "uploaded_exam.pdf")
            
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(base64.b64decode(pdf_data))
                pdf_path = tmp.name
            
            try:
                await mimic_exam_questions(
                    pdf_path=pdf_path,
                    kb_name=kb_name,
                    max_questions=max_questions,
                    ws_callback=ws_callback
                )
            finally:
                if Path(pdf_path).exists():
                    Path(pdf_path).unlink()
                    
        elif mode == "parsed":
            paper_path = data.get("paper_path")
            await mimic_exam_questions(
                paper_dir=paper_path,
                kb_name=kb_name,
                max_questions=max_questions,
                ws_callback=ws_callback
            )
        else:
            await websocket.send_json({"type": "error", "message": "Invalid mimic mode"})
            
        await websocket.send_json({"type": "complete", "message": "Mimic generation completed"})
            
    except WebSocketDisconnect:
        logger.info("Exam mimicry WebSocket disconnected")
    except Exception as e:
        logger.error(f"Exam mimicry error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass


@router.get("/status")
async def get_status():
    return {"status": "ok", "service": "Question Generation"}
