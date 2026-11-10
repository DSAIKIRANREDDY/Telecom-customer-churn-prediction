
import os
import yaml

from pathlib import Path
from box import ConfigBox


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read a YAML file and return its contents as a ConfigBox object.
    """

    try:
        with open(path_to_yaml, "r", encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)

            if content is None:
                content = {}

            return ConfigBox(content)

    except Exception as e:
        raise RuntimeError(
            f"Error while reading YAML file: {path_to_yaml}"
        ) from e


def create_directories(path_to_directories: list[Path]) -> None:
    """
    Create directories if they do not already exist.
    """

    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
