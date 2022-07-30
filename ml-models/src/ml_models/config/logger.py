import logging
import logging.config

import yaml


def get_logger(name: str = __name__) -> logging.Logger:
    with open('/app/ml_models/config/log_config.yml', 'r') as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    return logging.getLogger(name=name)
