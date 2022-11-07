from typing import Dict, Any
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config
