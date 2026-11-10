from telecom_churn.config.configuration import ConfigurationManager

config_manager = ConfigurationManager()

ingestion_config = config_manager.get_data_ingestion_config()
validation_config = config_manager.get_data_validation_config()
preprocessing_config = config_manager.get_data_preprocessing_config()
feature_config = config_manager.get_feature_engineering_config()
training_config = config_manager.get_model_training_config()
tuning_config = config_manager.get_model_tuning_config()
evaluation_config = config_manager.get_model_evaluation_config()
mlflow_config = config_manager.get_mlflow_config()

print("Ingestion:", ingestion_config)
print("Validation:", validation_config)
print("Preprocessing:", preprocessing_config)
print("Feature engineering:", feature_config)
print("Training:", training_config)
print("Tuning:", tuning_config)
print("Evaluation:", evaluation_config)
print("MLflow:", mlflow_config)