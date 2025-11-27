"""
FastAPI WebSocket API for FLOAT - Audio-Driven Talking Face Generation
Save this file as: api.py
Run with:cc
"""
import os
import json
import base64
import tempfile
import datetime
from pathlib import Path
from typing import Optional, Dict
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvloop

# Import the inference components
from generate import InferenceAgent, InferenceOptions

# Use uvloop for better performance
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

# Configuration
DEFAULT_IMAGE_PATH = "img.jpg"
RESULTS_DIR = "./results"
MAX_CONCURRENT_GENERATIONS = 3  # Maximum number of concurrent video generations

# Initialize FastAPI app
app = FastAPI(
    title="FLOAT WebSocket API",
    description="Audio-Driven Talking Face Generation via WebSocket and REST API",
    version="1.0.0"
)

# Pydantic models for POST request
class AudioData(BaseModel):
    content: str = Field(..., description="Base64 encoded audio data")
    ext: str = Field(..., description="Audio file extension (e.g., 'wav', 'mp3')")

class ImageData(BaseModel):
    content: str = Field(..., description="Base64 encoded image data")
    ext: str = Field(..., description="Image file extension (e.g., 'jpg', 'png')")

class GenerationParams(BaseModel):
    emotion: str = Field("S2E", description="Emotion control: S2E, angry, disgust, fear, happy, neutral, sad, surprise")
    a_cfg_scale: float = Field(2.0, ge=0.0, le=5.0, description="Audio CFG scale")
    r_cfg_scale: float = Field(1.0, ge=0.0, le=3.0, description="Reference CFG scale")
    e_cfg_scale: float = Field(1.0, ge=0.0, le=3.0, description="Emotion CFG scale")
    nfe: int = Field(10, ge=1, le=50, description="Number of function evaluations")
    seed: int = Field(25, ge=0, description="Random seed")
    no_crop: bool = Field(False, description="Skip face cropping")

class GenerationRequest(BaseModel):
    audio: AudioData
    image: Optional[ImageData] = None
    params: Optional[GenerationParams] = None

class BatchGenerationRequest(BaseModel):
    audios: list[AudioData] = Field(..., description="List of audio files to process")
    image: Optional[ImageData] = Field(None, description="Single image to use for all audios")
    images: Optional[list[ImageData]] = Field(None, description="List of images (one per audio)")
    params: Optional[GenerationParams] = Field(None, description="Generation parameters to use for all")
    
    class Config:
        schema_extra = {
            "example": {
                "audios": [
                    {"content": "base64_audio_1", "ext": "wav"},
                    {"content": "base64_audio_2", "ext": "wav"}
                ],
                "image": {"content": "base64_image", "ext": "jpg"},
                "params": {"emotion": "happy", "nfe": 10}
            }
        }

# Global model instance and semaphore
agent = None
opt = None
generation_semaphore = None  # Will be initialized on startup
active_generations = 0  # Track active generation count
generation_lock = asyncio.Lock()  # Lock for updating counter

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    global agent, opt, generation_semaphore
    
    print("\n" + "="*60)
    print("🚀 Starting FLOAT WebSocket API")
    print("="*60)
    
    # Initialize the semaphore for concurrent generation control
    generation_semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
    print(f"🔒 Concurrent generation limit: {MAX_CONCURRENT_GENERATIONS}")
    
    # Initialize options with empty args to use defaults
    import sys
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # Keep only script name
    
    opt = InferenceOptions().parse()
    opt.rank, opt.ngpus = 0, 1
    opt.res_dir = RESULTS_DIR
    
    sys.argv = original_argv
    
    # Create results directory
    os.makedirs(opt.res_dir, exist_ok=True)
    
    # Load model
    print("🤖 Loading FLOAT model...")
    agent = InferenceAgent(opt)
    print("✅ Model loaded successfully!")
    
    # Check default image
    if os.path.exists(DEFAULT_IMAGE_PATH):
        print(f"📸 Default image: {DEFAULT_IMAGE_PATH}")
    else:
        print(f"⚠️  Warning: Default image not found at {DEFAULT_IMAGE_PATH}")
    
    print(f"💾 Results directory: {opt.res_dir}")
    print("="*60)
    print("✅ Server ready! Waiting for connections...")
    print("="*60 + "\n")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "FLOAT WebSocket API",
        "version": "1.0.0",
        "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
        "active_generations": active_generations,
        "endpoints": {
            "websocket": "/ws",
            "rest_api": "/generate",
            "batch_api": "/generate/batch",
            "health": "/health",
            "status": "/status"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": agent is not None,
        "default_image_exists": os.path.exists(DEFAULT_IMAGE_PATH),
        "results_dir": opt.res_dir if opt else None,
        "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
        "active_generations": active_generations
    }

@app.get("/status")
async def get_status():
    """Get current generation status"""
    available_slots = MAX_CONCURRENT_GENERATIONS - active_generations
    return {
        "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS,
        "active_generations": active_generations,
        "available_slots": available_slots,
        "queue_full": available_slots == 0
    }

@app.post("/generate")
async def generate_video(request: GenerationRequest):
    """
    REST API endpoint for video generation
    
    Request body:
    {
        "audio": {"content": "base64_data", "ext": "wav"},
        "image": {"content": "base64_data", "ext": "jpg"},  // Optional
        "params": {
            "emotion": "S2E",
            "a_cfg_scale": 2.0,
            "r_cfg_scale": 1.0,
            "e_cfg_scale": 1.0,
            "nfe": 10,
            "seed": 25,
            "no_crop": false
        }
    }
    
    Returns:
    {
        "status": "success",
        "video": "base64_encoded_video",
        "video_path": "/path/to/video.mp4",
        "message": "Video generated successfully",
        "params_used": {...}
    }
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Convert Pydantic models to dict format expected by process_generation_request
        request_data = {
            "audio": {
                "content": request.audio.content,
                "ext": request.audio.ext
            }
        }
        
        # Add image if provided
        if request.image:
            request_data["image"] = {
                "content": request.image.content,
                "ext": request.image.ext
            }
        
        # Add params if provided
        if request.params:
            request_data["params"] = {
                "emotion": request.params.emotion,
                "a_cfg_scale": request.params.a_cfg_scale,
                "r_cfg_scale": request.params.r_cfg_scale,
                "e_cfg_scale": request.params.e_cfg_scale,
                "nfe": request.params.nfe,
                "seed": request.params.seed,
                "no_crop": request.params.no_crop
            }
        
        # Process the request with semaphore control
        result = await process_generation_request(request_data)
        
        # Return error if processing failed
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error", "Generation failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate/batch")
async def generate_video_batch(request: BatchGenerationRequest):
    """
    Batch video generation endpoint - process multiple audios at once
    
    Request body:
    {
        "audios": [
            {"content": "base64_audio_1", "ext": "wav"},
            {"content": "base64_audio_2", "ext": "wav"}
        ],
        "image": {"content": "base64_image", "ext": "jpg"},  // Optional: single image for all
        "images": [  // Optional: one image per audio (overrides "image")
            {"content": "base64_image_1", "ext": "jpg"},
            {"content": "base64_image_2", "ext": "jpg"}
        ],
        "params": {
            "emotion": "S2E",
            "nfe": 10,
            ...
        }
    }
    
    Returns:
    {
        "status": "success",
        "total": 2,
        "successful": 2,
        "failed": 0,
        "results": [
            {
                "index": 0,
                "status": "success",
                "video": "base64_video_1",
                "video_path": "/path/to/video1.mp4",
                ...
            },
            {
                "index": 1,
                "status": "success",
                "video": "base64_video_2",
                "video_path": "/path/to/video2.mp4",
                ...
            }
        ]
    }
    """
    if agent is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Validate input
        num_audios = len(request.audios)
        if num_audios == 0:
            raise HTTPException(status_code=400, detail="At least one audio file is required")
        
        # Check if using per-audio images or single image
        if request.images:
            if len(request.images) != num_audios:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Number of images ({len(request.images)}) must match number of audios ({num_audios})"
                )
            use_multiple_images = True
        else:
            use_multiple_images = False
        
        print(f"\n📦 Processing batch of {num_audios} audio files...")
        print(f"🔒 Concurrent limit: {MAX_CONCURRENT_GENERATIONS}, Active: {active_generations}")
        
        # Process each audio (semaphore will automatically control concurrency)
        tasks = []
        
        for idx, audio in enumerate(request.audios):
            # Build request for this audio
            single_request = {
                "audio": {
                    "content": audio.content,
                    "ext": audio.ext
                }
            }
            
            # Determine which image to use
            if use_multiple_images:
                single_request["image"] = {
                    "content": request.images[idx].content,
                    "ext": request.images[idx].ext
                }
            elif request.image:
                single_request["image"] = {
                    "content": request.image.content,
                    "ext": request.image.ext
                }
            
            # Add params if provided
            if request.params:
                single_request["params"] = {
                    "emotion": request.params.emotion,
                    "a_cfg_scale": request.params.a_cfg_scale,
                    "r_cfg_scale": request.params.r_cfg_scale,
                    "e_cfg_scale": request.params.e_cfg_scale,
                    "nfe": request.params.nfe,
                    "seed": request.params.seed,
                    "no_crop": request.params.no_crop
                }
            
            # Create task for this generation
            task = process_single_batch_item(idx, single_request, num_audios)
            tasks.append(task)
        
        # Wait for all tasks to complete (concurrency controlled by semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful = 0
        failed = 0
        processed_results = []
        
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                failed += 1
                processed_results.append({
                    "index": idx,
                    "status": "error",
                    "error": str(result),
                    "message": f"Failed to process audio {idx + 1}"
                })
            else:
                if result.get("status") == "success":
                    successful += 1
                else:
                    failed += 1
                processed_results.append(result)
        
        print(f"\n📊 Batch processing complete: {successful} successful, {failed} failed")
        
        # Return batch results
        return {
            "status": "success" if failed == 0 else "partial",
            "total": num_audios,
            "successful": successful,
            "failed": failed,
            "results": processed_results,
            "message": f"Processed {num_audios} audios: {successful} successful, {failed} failed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing error: {str(e)}")

async def process_single_batch_item(idx: int, request_data: Dict, total: int) -> Dict:
    """Process a single batch item with logging"""
    print(f"\n🎵 Queuing audio {idx + 1}/{total}...")
    
    try:
        result = await process_generation_request(request_data)
        result["index"] = idx
        
        if result.get("status") == "success":
            print(f"✅ Audio {idx + 1} completed successfully")
        else:
            print(f"❌ Audio {idx + 1} failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        print(f"❌ Audio {idx + 1} failed with exception: {str(e)}")
        return {
            "index": idx,
            "status": "error",
            "error": str(e),
            "message": f"Failed to process audio {idx + 1}"
        }

def save_base64_file(base64_content: str, extension: str, prefix: str = "temp") -> str:
    """Save base64 content to a temporary file"""
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, 
        suffix=f".{extension}", 
        prefix=f"{prefix}_"
    )
    
    file_data = base64.b64decode(base64_content)
    temp_file.write(file_data)
    temp_file.close()
    
    return temp_file.name

def read_file_as_base64(file_path: str) -> str:
    """Read file and return as base64 string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')

async def process_generation_request(request_data: Dict) -> Dict:
    """
    Process video generation request with semaphore-based concurrency control
    
    Expected format:
    {
        "audio": {"content": "base64_data", "ext": "wav"},
        "image": {"content": "base64_data", "ext": "jpg"},  # Optional
        "params": {
            "emotion": "S2E",
            "a_cfg_scale": 2.0,
            "r_cfg_scale": 1.0,
            "e_cfg_scale": 1.0,
            "nfe": 10,
            "seed": 25,
            "no_crop": false
        }
    }
    """
    global active_generations
    
    # Acquire semaphore to limit concurrent generations
    async with generation_semaphore:
        # Update active generation counter
        async with generation_lock:
            active_generations += 1
            current_count = active_generations
        
        print(f"🔒 Generation slot acquired ({current_count}/{MAX_CONCURRENT_GENERATIONS} active)")
        
        temp_files = []
        
        try:
            # Validate audio
            if "audio" not in request_data:
                return {
                    "status": "error",
                    "error": "Missing 'audio' field in request"
                }
            
            audio_data = request_data["audio"]
            if "content" not in audio_data or "ext" not in audio_data:
                return {
                    "status": "error",
                    "error": "Audio must have 'content' and 'ext' fields"
                }
            
            # Save audio to temp file
            audio_ext = audio_data["ext"].lstrip('.')
            audio_path = save_base64_file(audio_data["content"], audio_ext, "audio")
            temp_files.append(audio_path)
            print(f"💾 Saved audio: {audio_path}")
            
            # Handle image
            if "image" in request_data and request_data["image"]:
                image_data = request_data["image"]
                if "content" not in image_data or "ext" not in image_data:
                    return {
                        "status": "error",
                        "error": "Image must have 'content' and 'ext' fields"
                    }
                
                image_ext = image_data["ext"].lstrip('.')
                image_path = save_base64_file(image_data["content"], image_ext, "image")
                temp_files.append(image_path)
                print(f"📸 Using uploaded image: {image_path}")
            else:
                if not os.path.exists(DEFAULT_IMAGE_PATH):
                    return {
                        "status": "error",
                        "error": f"Default image not found: {DEFAULT_IMAGE_PATH}"
                    }
                image_path = DEFAULT_IMAGE_PATH
                print(f"📸 Using default image: {image_path}")
            
            # Extract parameters
            params = request_data.get("params", {})
            emotion = params.get("emotion", "S2E")
            a_cfg_scale = params.get("a_cfg_scale", 2.0)
            r_cfg_scale = params.get("r_cfg_scale", 1.0)
            e_cfg_scale = params.get("e_cfg_scale", 1.0)
            nfe = params.get("nfe", 10)
            seed = params.get("seed", 25)
            no_crop = params.get("no_crop", False)
            
            # Generate output path
            call_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            res_video_path = os.path.join(
                opt.res_dir,
                f"{call_time}_generated_nfe{nfe}_seed{seed}.mp4"
            )
            
            print(f"🎬 Generating video...")
            print(f"   Emotion: {emotion}, NFE: {nfe}, Seed: {seed}")
            
            # Run inference in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            output_path = await loop.run_in_executor(
                None,
                lambda: agent.run_inference(
                    res_video_path=res_video_path,
                    ref_path=image_path,
                    audio_path=audio_path,
                    a_cfg_scale=a_cfg_scale,
                    r_cfg_scale=r_cfg_scale,
                    e_cfg_scale=e_cfg_scale,
                    emo=emotion,
                    nfe=nfe,
                    no_crop=no_crop,
                    seed=seed,
                    verbose=True
                )
            )
            
            print(f"✅ Video generated: {output_path}")
            
            # Read video as base64
            video_base64 = read_file_as_base64(output_path)
            
            return {
                "status": "success",
                "video": video_base64,
                "video_path": output_path,
                "message": "Video generated successfully",
                "params_used": {
                    "emotion": emotion,
                    "nfe": nfe,
                    "seed": seed,
                    "a_cfg_scale": a_cfg_scale,
                    "r_cfg_scale": r_cfg_scale,
                    "e_cfg_scale": e_cfg_scale
                }
            }
            
        except Exception as e:
            print(f"❌ Error during processing: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "status": "error",
                "error": str(e),
                "message": "Failed to generate video"
            }
        
        finally:
            # Cleanup temp files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        print(f"🗑️  Cleaned up: {temp_file}")
                except Exception as e:
                    print(f"⚠️  Failed to delete {temp_file}: {e}")
            
            # Release generation slot
            async with generation_lock:
                active_generations -= 1
                remaining = active_generations
            
            print(f"🔓 Generation slot released ({remaining}/{MAX_CONCURRENT_GENERATIONS} active)")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for video generation"""
    await websocket.accept()
    
    client_id = f"{websocket.client.host}:{websocket.client.port}"
    print(f"\n🔌 New connection from {client_id}")
    
    try:
        # Send welcome message
        await websocket.send_json({
            "status": "connected",
            "message": "Connected to FLOAT WebSocket API",
            "version": "1.0.0",
            "max_concurrent_generations": MAX_CONCURRENT_GENERATIONS
        })
        
        # Process messages
        while True:
            try:
                # Receive message
                message = await websocket.receive_text()
                print(f"📨 Received message from {client_id}")
                
                # Parse JSON
                request_data = json.loads(message)
                
                # Send processing acknowledgment
                await websocket.send_json({
                    "status": "processing",
                    "message": "Processing your request...",
                    "active_generations": active_generations,
                    "max_concurrent": MAX_CONCURRENT_GENERATIONS
                })
                
                # Process request (semaphore will control concurrency)
                response = await process_generation_request(request_data)
                
                # Send response
                await websocket.send_json(response)
                print(f"✉️  Sent response to {client_id}")
                
            except json.JSONDecodeError as e:
                await websocket.send_json({
                    "status": "error",
                    "error": f"Invalid JSON: {str(e)}"
                })
                print(f"⚠️  JSON decode error from {client_id}")
            
            except Exception as e:
                await websocket.send_json({
                    "status": "error",
                    "error": f"Processing error: {str(e)}"
                })
                print(f"❌ Error handling message from {client_id}: {e}")
    
    except WebSocketDisconnect:
        print(f"🔌 Client disconnected: {client_id}")
    
    except Exception as e:
        print(f"❌ WebSocket error with {client_id}: {e}")
    
    finally:
        print(f"👋 Connection closed: {client_id}\n")

# For running with uvicorn (alternative to hypercorn)
"""if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )"""