import signal
import sys
from core.deployment import ManagedGPU
from core.client import request_image_generation

def main():
    """
    Main interactive loop to launch, manage, and automatically terminate a GPU server.
    """
    # Define a signal handler for Ctrl+C
    def signal_handler(sig, frame):
        print("\nCtrl+C detected. Shutting down gracefully...")
        # The __exit__ method of the ManagedGPU context manager will handle termination.
        sys.exit(0)

    # Register the signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Use the context manager to ensure cleanup
        with ManagedGPU() as gpu:
            print(f"\n\nServer is ready at {gpu.ip_address}. Starting interactive session.")
            print("Type 'quit' or 'exit' at the prompt to terminate.")

            # Interactive image generation loop
            while True:
                prompt = input("\nEnter your prompt: ")
                if prompt.lower() in ['quit', 'exit']:
                    break

                while True:
                    try:
                        num_images_str = input("How many images (1-4)? ")
                        if num_images_str.lower() in ['quit', 'exit']:
                            prompt = 'quit'  # To exit outer loop
                            break
                        num_images = int(num_images_str)
                        if 1 <= num_images <= 4:
                            break
                        else:
                            print("Please enter a number between 1 and 4.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
                
                if prompt.lower() in ['quit', 'exit']:
                    break

                request_image_generation(gpu.ip_address, prompt, num_images)

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        # The __exit__ method will still be called for cleanup
    
    print("\nSession finished. The GPU instance has been terminated.")

if __name__ == "__main__":
    main()
