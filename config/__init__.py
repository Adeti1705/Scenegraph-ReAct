import os
import yaml
import argparse
from pathlib import Path


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_config(config: dict, args: argparse.Namespace):
    defaults = config.copy()
    for key, value in args.__dict__.items():
        if key in config and value is not None:
            config[key] = value
    for key in ("api_base", "api_key", "api_version"):
        value = config[key]
        cli_value = getattr(args, key, None)
        if cli_value is not None and cli_value != defaults.get(key):
            config[key] = os.getenv(value, value)
        else:
            config[key] = os.getenv(value)
    return config


# General config
CONFIG_DIR = Path(__file__).resolve().parent

general_config = load_config(str(CONFIG_DIR / "general.yaml"))

prompts_config = load_config(str(CONFIG_DIR / "prompts.yaml"))