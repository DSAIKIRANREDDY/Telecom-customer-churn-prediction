
import pandas as pd
import joblib
from pathlib import Path

from telecom_churn.inference.model_loader import ModelLoader
from telecom_churn.inference.explainability import CustomerExplainability


MODEL_PATH = Path(
    "artifacts/models/tuned/xgboost_tuned_model.pkl"
)

EXPLANATION_DIR = Path(
    "artifacts/reports/explainability/inference"
)

DATA_PATH = Path(
    "test_customers.csv"
)


# Load model and explainer
model_loader = ModelLoader(MODEL_PATH)

model_loader.load_model()
model_loader.load_explainer()


# Initialize explainability service
customer_explainer = CustomerExplainability(
    model_loader=model_loader,
    output_dir=EXPLANATION_DIR
)


# Load validation data
customers_df = pd.read_csv(DATA_PATH)

# Use only five customers for the initial test
customers_df = customers_df.head(5)


# Generate batch SHAP explanation
result = customer_explainer.explain_batch(
    customers_df
)


print("\nTotal customers:", result["total_customers"])

print("\nFeature importance:")
print(result["feature_importance"])

print("\nBeeswarm plot:")
print(result["beeswarm_plot_path"])

print("\nFeature importance plot:")
print(result["feature_importance_plot_path"])
