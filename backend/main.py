import os
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv()

# Import our custom processing modules
from .document_processor import extract_text_from_bytes
from .question_generator import generate_questions
from .groq_generator import (
    generate_questions_groq,
    check_groq_connection,
    get_groq_api_keys,
    GROQ_MODELS,
)

app = FastAPI(
    title="MathGenix API",
    description="Backend API for generating math questions using local Ollama or Groq cloud LPU inference",
    version="3.0.0"
)

# Enable CORS to allow the React frontend to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the exact origin of the frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_API_URL = os.environ.get("OLLAMA_API_URL", "http://127.0.0.1:11434")

class GenerateRequest(BaseModel):
    text: str
    taxonomy_level: str
    model_name: str = "llama3"
    provider: str = "groq"  # "local" or "groq"
    previous_topics: List[str] = []  # Topics from prior generations to avoid repeats

class ApiKeyRequest(BaseModel):
    api_key: str

@app.get("/api/health")
async def health_check():
    """
    Checks the status of the FastAPI backend, local Ollama, and Groq cloud connections.
    """
    # Check Ollama
    ollama_status = "Disconnected"
    ollama_available = False
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(OLLAMA_API_URL)
            if response.status_code == 200:
                ollama_status = "Connected"
                ollama_available = True
    except Exception:
        ollama_status = "Disconnected (Ensure Ollama is running locally)"

    # Check Groq
    groq_keys = get_groq_api_keys()
    # Check the first key for basic connectivity status
    groq_info = check_groq_connection(groq_keys[0]) if groq_keys else {"status": "No API Key", "available": False}

    return {
        "status": "healthy",
        "ollama": {
            "status": ollama_status,
            "url": OLLAMA_API_URL,
            "available": ollama_available
        },
        "groq": {
            "status": groq_info["status"],
            "available": groq_info["available"],
            "has_key": len(groq_keys) > 0,
            "key_count": len(groq_keys)
        }
    }

@app.get("/api/debug-keys")
async def debug_keys():
    """Temporary debug endpoint — shows how many Groq keys are loaded."""
    keys = get_groq_api_keys()
    return {
        "key_count": len(keys),
        "keys_preview": [f"...{k[-4:]}" for k in keys],
        "env_GROQ_API_KEYS": bool(os.environ.get("GROQ_API_KEYS")),
        "env_GROQ_API_KEY": bool(os.environ.get("GROQ_API_KEY")),
    }

@app.get("/api/models")
async def list_models():
    """
    Queries the local Ollama instance for pulled models, falling back to a static list if offline.
    """
    fallback_models = ["qwen2-math", "deepseek-r1", "wizard-math", "llama3", "mistral"]
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{OLLAMA_API_URL}/api/tags")
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                if models:
                    return {"models": models, "source": "local_ollama"}
    except Exception:
        pass
    
    return {"models": fallback_models, "source": "static_fallback"}

@app.get("/api/groq-models")
async def list_groq_models():
    """
    Returns the list of available Groq cloud models.
    """
    return {"models": GROQ_MODELS}

@app.post("/api/set-groq-key")
async def set_groq_key(request: ApiKeyRequest):
    """
    Sets the Groq API key in the environment and persists it to .env file.
    """
    api_key = request.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    
    # Set in current process environment
    os.environ["GROQ_API_KEYS"] = api_key
    
    # Persist to .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    try:
        # Read existing .env content
        existing_lines = []
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                existing_lines = f.readlines()
        
        # Update or append GROQ_API_KEYS (and remove old GROQ_API_KEY)
        key_found = False
        new_lines = []
        for line in existing_lines:
            if line.strip().startswith("GROQ_API_KEYS=") or line.strip().startswith("GROQ_API_KEY="):
                if not key_found:
                    new_lines.append(f"GROQ_API_KEYS={api_key}\n")
                    key_found = True
            else:
                new_lines.append(line)
        
        if not key_found:
            new_lines.append(f"GROQ_API_KEYS={api_key}\n")
        
        with open(env_path, "w") as f:
            f.writelines(new_lines)
        
    except Exception as e:
        print(f"Warning: Could not persist API key to .env: {e}")
    
    # Verify the key works (check the first one)
    keys_list = [k.strip() for k in api_key.split(",")]
    status = check_groq_connection(keys_list[0]) if keys_list else {"status": "No API Key", "available": False}
    
    return {
        "success": True,
        "groq_status": status["status"],
        "groq_available": status["available"],
    }

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Handles file upload, extracts text in memory, and returns the raw text content.
    For image files and scanned PDFs, uses Groq Vision OCR to extract text.
    """
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    try:
        file_bytes = await file.read()

        # Pass Groq API key for vision OCR (images & scanned PDFs)
        groq_keys = get_groq_api_keys()
        primary_key = groq_keys[0] if groq_keys else ""
        result = extract_text_from_bytes(file_bytes, filename, api_key=primary_key)

        text_content = result["text"]
        ocr_used = result["ocr_used"]

        if not text_content or not text_content.strip():
            raise HTTPException(status_code=400, detail="Document appears to be empty or contains no readable text.")
            
        return {
            "filename": filename,
            "char_count": len(text_content),
            "word_count": len(text_content.split()),
            "text": text_content,
            "ocr_used": ocr_used,
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

@app.post("/api/generate")
async def generate_questions_endpoint(request: GenerateRequest):
    """
    Generates structured math questions using either local Ollama or Groq cloud.
    Routes to the appropriate generator based on the 'provider' field.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text context is empty.")
    
    if request.provider == "groq":
        # Cloud generation via Groq LPU
        api_keys = get_groq_api_keys()
        result = generate_questions_groq(
            text=request.text,
            taxonomy_level=request.taxonomy_level,
            model_name=request.model_name,
            api_keys=api_keys,
            previous_topics=request.previous_topics,
        )
    else:
        # Local generation via Ollama
        result = generate_questions(
            text=request.text,
            taxonomy_level=request.taxonomy_level,
            model_name=request.model_name,
            previous_topics=request.previous_topics,
        )
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
        
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
