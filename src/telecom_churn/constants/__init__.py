
from pathlib import Path


# Absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Configuration files
CONFIG_FILE_PATH = PROJECT_ROOT / "config" / "config.yaml"

SCHEMA_FILE_PATH = PROJECT_ROOT / "config" / "schema.yaml"
