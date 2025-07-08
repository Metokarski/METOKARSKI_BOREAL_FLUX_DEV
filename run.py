import signal
import sys
from core.deployment import ManagedGPU
from core.client import request_image_generation
from core.logger import get_logger

log = get_logger(__name__)

def main():
    """
    Main interactive loop to launch, manage, and automatically terminate a GPU server.
    """
    # Define a signal handler for Ctrl+C
    def signal_handler(sig, frame):
        log.info("Ctrl+C detected. Shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    log.info("--- Starting Boreal Flux Application ---")
    try:
        with ManagedGPU() as gpu:
            if not gpu.ip_address:
                log.critical("Failed to acquire GPU instance IP address. Aborting.")
                return

            log.info(f"Server is ready at {gpu.ip_address}. Starting interactive session.")
            print("\nType 'quit' or 'exit' at the prompt to terminate.")

            while True:
                try:
                    prompt = input("\nEnter your prompt: ")
                    if prompt.lower() in ['quit', 'exit']:
                        log.info("User requested to quit.")
                        break

                    while True:
                        try:
                            num_images_str = input("How many images (1-4)? ")
                            if num_images_str.lower() in ['quit', 'exit']:
                                prompt = 'quit'
                                break
                            num_images = int(num_images_str)
                            if 1 <= num_images <= 4:
                                break
                            else:
                                print("Please enter a number between 1 and 4.")
                        except ValueError:
                            log.warning("Invalid input for number of images.")
                            print("Invalid input. Please enter a number.")
                    
                    if prompt.lower() in ['quit', 'exit']:
                        log.info("User requested to quit.")
                        break

                    request_image_generation(gpu.ip_address, prompt, num_images)
                
                except (EOFError, KeyboardInterrupt):
                    log.info("User interrupted the session. Exiting.")
                    break

    except Exception as e:
        log.critical(f"A critical error occurred in the main loop: {e}", exc_info=True)
    
    log.info("--- Boreal Flux Application Finished ---")

if __name__ == "__main__":
    main()
