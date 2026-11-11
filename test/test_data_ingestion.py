
from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.data_ingestion import DataIngestion


# Get configuration
config_manager = ConfigurationManager()

data_ingestion_config = (
    config_manager.get_data_ingestion_config()
)

# Initialize data ingestion component
data_ingestion = DataIngestion(
    config=data_ingestion_config
)

# Execute data ingestion
ingested_file = data_ingestion.initiate_data_ingestion()

print("\nIngested dataset:", ingested_file)
