import requests
import os

def request_image_generation(server_ip: str, prompt: str, num_images: int = 1):
    """
    Sends a request to the server to generate images and downloads them.
    """
    server_url = f"http://{server_ip}:8000"
    generate_url = f"{server_url}/generate/"
    images_url = f"{server_url}/images/"

    print(f"\n--- Requesting {num_images} image(s) for prompt: '{prompt}' ---")

    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images}...")
        payload = {"prompt": prompt}
        
        try:
            response = requests.post(generate_url, json=payload, timeout=300) # 5-minute timeout for generation
            response.raise_for_status()
            filename = response.json()['filename']
            print(f"  > Generation complete. Filename: {filename}")

            # Download the image
            print(f"  > Downloading {filename}...")
            image_response = requests.get(f"{images_url}{filename}", stream=True)
            image_response.raise_for_status()

            local_save_dir = "generated_images"
            os.makedirs(local_save_dir, exist_ok=True)
            local_filepath = os.path.join(local_save_dir, filename)

            with open(local_filepath, 'wb') as f:
                for chunk in image_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"  > Image saved successfully to {local_filepath}")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            break # Stop if one image fails
