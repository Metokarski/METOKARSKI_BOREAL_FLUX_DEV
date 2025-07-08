import os
import torch
from diffusers import FluxPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import datetime
import uuid

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Model Loading ---
# This section is time-consuming, so it's done once when the server starts.
HUGGING_FACE_TOKEN = os.environ.get("HUGGING_FACE_TOKEN")
if not HUGGING_FACE_TOKEN:
    raise ValueError("HUGGING_FACE_TOKEN environment variable not set. The server cannot start.")

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    safety_checker=None,
    use_auth_token=HUGGING_FACE_TOKEN
)
pipe.load_lora_weights("kudzueye/boreal-flux-dev-v2", weight_name="boreal-v2.safetensors")
pipe.fuse_lora(lora_scale=1.0)
pipe.enable_model_cpu_offload()

# --- Image Saving Configuration ---
SAVE_DIRECTORY = "generated_images"
os.makedirs(SAVE_DIRECTORY, exist_ok=True)

def generate_filename(prompt: str) -> str:
    """Generates a descriptive filename for the image."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Sanitize prompt for filename
    safe_prompt = "".join([c if c.isalnum() else "_" for c in prompt.split()[:5]]).rstrip('_')
    unique_id = str(uuid.uuid4())[:8]
    return f"{timestamp}_{safe_prompt}_{unique_id}.png"

# --- API Models ---
class GenerationRequest(BaseModel):
    prompt: str
    height: int = 1024
    width: int = 1024
    num_inference_steps: int = 50
    guidance_scale: float = 7.5

# --- API Endpoints ---
@app.post("/generate/")
async def generate_image(request: GenerationRequest):
    """
    Generates an image based on the provided prompt and parameters.
    Returns the filename of the generated image.
    """
    try:
        generator = torch.Generator(device="cpu").manual_seed(int(uuid.uuid4().int & (1<<32)-1))
        
        image = pipe(
            prompt=request.prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            generator=generator
        ).images[0]

        filename = generate_filename(request.prompt)
        filepath = os.path.join(SAVE_DIRECTORY, filename)
        image.save(filepath)

        return {"filename": filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images/{filename}")
async def get_image(filename: str):
    """
    Serves a generated image file.
    """
    filepath = os.path.join(SAVE_DIRECTORY, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

@app.get("/")
def read_root():
    return {"message": "Inference server is running. Use /generate to create images."}

# To run this server: uvicorn inference_server:app --host 0.0.0.0 --port 8000
