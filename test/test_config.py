from telecom_churn.constants import CONFIG_FILE_PATH, PROJECT_ROOT
from telecom_churn.utils.common import read_yaml, create_directories

# Read YAML configuration
config = read_yaml(CONFIG_FILE_PATH)

print(config.data_ingestion.source_data_path)
print(config.model_training.algorithm)

# Create directories
create_directories([
    PROJECT_ROOT / "data" / "interim",
    PROJECT_ROOT / "data" / "validated",
    PROJECT_ROOT / "artifacts" / "preprocessors"
])

print("Configuration loaded successfully!")
print("Directories created successfully!")