import logging
import sys

def get_logger(name):
    """
    Configures and returns a logger with a standardized format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        
        # Create a handler for stdout
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.DEBUG)
        
        # Create a handler for a log file
        file_handler = logging.FileHandler('boreal_flux.log')
        file_handler.setLevel(logging.DEBUG)
        
        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        stdout_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(stdout_handler)
        logger.addHandler(file_handler)
        
    return logger
