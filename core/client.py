import requests
import os
from core.logger import get_logger

log = get_logger(__name__)

def request_image_generation(server_ip: str, prompt: str, num_images: int = 1):
    """
    Sends a request to the server to generate images and downloads them.
    """
    server_url = f"http://{server_ip}:8000"
    generate_url = f"{server_url}/generate/"
    images_url = f"{server_url}/images/"

    log.info(f"--- Requesting {num_images} image(s) for prompt: '{prompt}' ---")

    for i in range(num_images):
        log.info(f"Generating image {i+1}/{num_images}...")
        payload = {"prompt": prompt}
        
        try:
            log.debug(f"Sending POST request to {generate_url} with payload: {payload}")
            response = requests.post(generate_url, json=payload, timeout=300)
            response.raise_for_status()
            
            filename = response.json()['filename']
            log.info(f"Generation successful. Server returned filename: {filename}")

            log.info(f"Downloading {filename} from {images_url}{filename}...")
            image_response = requests.get(f"{images_url}{filename}", stream=True)
            image_response.raise_for_status()

            local_save_dir = "generated_images"
            os.makedirs(local_save_dir, exist_ok=True)
            local_filepath = os.path.join(local_save_dir, filename)

            with open(local_filepath, 'wb') as f:
                for chunk in image_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            log.info(f"Image saved successfully to {local_filepath}")

        except requests.exceptions.Timeout:
            log.error("The request to the generation server timed out.", exc_info=True)
            break
        except requests.exceptions.RequestException as e:
            log.error(f"An error occurred during the request to the server: {e}", exc_info=True)
            break
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}", exc_info=True)
            break
