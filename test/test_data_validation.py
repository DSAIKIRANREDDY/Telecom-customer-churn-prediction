
from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.data_validation import DataValidation


# Initialize configuration manager
config_manager = ConfigurationManager()

# Get validation configuration
validation_config = (
    config_manager.get_data_validation_config()
)

# Initialize data validation component
data_validation = DataValidation(
    config=validation_config
)

# Execute validation
validated_file = (
    data_validation.initiate_data_validation()
)

print("\nValidated dataset:", validated_file)

