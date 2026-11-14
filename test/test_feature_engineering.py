
from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.feature_engineering import FeatureEngineering


config = ConfigurationManager().get_feature_engineering_config()

feature_engineering = FeatureEngineering(config)

output_paths = feature_engineering.initiate_feature_engineering()

print("\nFeature Engineering Output:")

for name, path in output_paths.items():
    print(f"{name} -> {path}")

import pandas as pd
df = pd.read_csv("data/featured/train_featured.csv")

print("Shape:", df.shape)
print("TotalCharges dtype:", df["TotalCharges"].dtype)
print("Missing TotalCharges:", df["TotalCharges"].isna().sum())