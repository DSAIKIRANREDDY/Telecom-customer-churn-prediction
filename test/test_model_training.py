
from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.model_training import ModelTraining


config = ConfigurationManager().get_model_training_config()

model_training = ModelTraining(config)

model_training.initiate_model_training()


import joblib

model = joblib.load(
    "artifacts/models/trained/xgboost_model.pkl"
)

preprocessor = model.named_steps["preprocessor"]

print("Transformed feature count:",
      len(preprocessor.get_feature_names_out()))

print("Numerical features:",
      len(preprocessor.transformers_[0][2]))

print("Categorical features:",
      len(preprocessor.transformers_[1][2]))