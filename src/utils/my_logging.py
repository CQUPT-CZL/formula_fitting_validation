import logging
import os


def setup_logging(config: dict):
    """Set up logging with file output."""
    log_file = config['logging']['file']
    log_level = getattr(logging, config['logging']['level'].upper(), logging.INFO)

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized")