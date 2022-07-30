import logging
import logging.config

import yaml

from ml_models.config.constants import LOGGER_CONFIG_FILE


def get_logger(name: str = __name__) -> logging.Logger:
    with open(LOGGER_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    return logging.getLogger(name=name)
