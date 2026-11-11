
from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.model_training import ModelTraining


config = ConfigurationManager().get_model_training_config()

model_training = ModelTraining(config)

model_training.initiate_model_training()
