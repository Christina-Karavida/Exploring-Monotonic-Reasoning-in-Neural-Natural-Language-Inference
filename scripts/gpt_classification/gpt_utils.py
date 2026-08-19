import os
import logging
from datetime import datetime


def get_logger():
    # Configure logging
    log_dir = "./logs"
    os.makedirs(log_dir, exist_ok=True)

    # Get the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"process_and_train_{current_time}.log")
    print("Logging to:", log_file_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.StreamHandler(),
                            logging.FileHandler(log_file_path)
                        ])

    logger = logging.getLogger(__name__)
    return logger