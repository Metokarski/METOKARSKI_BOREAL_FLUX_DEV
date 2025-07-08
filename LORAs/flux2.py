import os
import torch
from diffusers import FluxPipeline

# Set your Hugging Face access token
HUGGING_FACE_TOKEN = "REMOVED"

# Function to get the next image filename
def get_next_image_filename(directory: str, base_name: str, extension: str) -> str:
    existing_files = os.listdir(directory)
    # Count images with similar base names to find the next number
    count = sum(1 for file in existing_files if file.startswith(base_name) and file.endswith(extension))
    # Create the new filename
    return os.path.join(directory, f'{base_name}_{count + 1}{extension}')

# Define the directory to save images
save_directory = 'generated_images'

# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load the base FLUX model with authentication
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,
    safety_checker=None,  # Disable the NSFW filter
    use_auth_token=HUGGING_FACE_TOKEN  # Use the access token for authentication
)

# Load and fuse the LoRA weights using diffusers' built-in methods
pipe.load_lora_weights("kudzueye/boreal-flux-dev-v2", weight_name="boreal-v2.safetensors")
pipe.fuse_lora(lora_scale=1.0)

# Optional: Offload to CPU to save VRAM if needed
pipe.enable_model_cpu_offload()

# Generate multiple images using different seeds for each image
prompt = "elizabeth moss sits on a couch wearing sweatpants and a bra on halloween night"
num_images = 5  # Number of images to generate

# Create a list of generators with different seeds
generators = [torch.Generator(device="cpu").manual_seed(i) for i in range(num_images)]

images = pipe(
    prompt=prompt,
    height=1632,
    width=3840,
    guidance_scale=3.5,
    num_inference_steps=100,
    max_sequence_length=512,
    num_images_per_prompt=num_images,
    generator=generators  # Pass the list of generators
).images

# Save each generated image with an incrementing filename
for i, image in enumerate(images):
    image_filename = get_next_image_filename(save_directory, 'flux-realism', '.png')
    image.save(image_filename)
