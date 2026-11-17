
import pandas as pd

from pathlib import Path

from telecom_churn.inference.model_loader import ModelLoader
from telecom_churn.inference.predictor import ChurnPredictor
from telecom_churn.inference.explainability import CustomerExplainability

# --------------------------------------------------
# 1. Load model
# --------------------------------------------------

model_path = Path(
    "artifacts/models/tuned/xgboost_tuned_model.pkl"
)

model_loader = ModelLoader(model_path)

model_loader.load_model()

# Initialize SHAP explainer for later use
model_loader.load_explainer()


# --------------------------------------------------
# 2. Initialize predictor
# --------------------------------------------------

predictor = ChurnPredictor(
    model_loader=model_loader,
    threshold=0.65
)

# Initialize customer explainability
customer_explainer = CustomerExplainability(
    model_loader=model_loader
)

# --------------------------------------------------
# 3. Load validation data for testing
# --------------------------------------------------

validation_path = Path(
    "data/featured/validation_featured.csv"
)

validation_df = pd.read_csv(validation_path)

# Select original input features
input_columns = (
    ["customerID"] + predictor.expected_features
)

test_data = validation_df[input_columns].copy()


# --------------------------------------------------
# 4. Test single customer prediction
# --------------------------------------------------

print("\n========== SINGLE CUSTOMER PREDICTION ==========")

customer = test_data.iloc[0].to_dict()

single_result = predictor.predict_single(customer)

print(single_result)

# Test customer-level SHAP explanation

print("\n========== CUSTOMER SHAP EXPLANATION ==========")

customer_explanation = customer_explainer.explain_customer(
    customer
)

print("\nCustomer ID:")
print(customer_explanation["customerID"])

print("\nBase value:")
print(customer_explanation["base_value"])

print("\nTop customer contributions:")
print(
    customer_explanation["contributions"].head(10)
)

print("\nWaterfall plot:")
print(customer_explanation["plot_path"])


# --------------------------------------------------
# 5. Test batch prediction
# --------------------------------------------------

print("\n========== BATCH PREDICTION ==========")

batch_data = test_data.head(5)

batch_results = predictor.predict_batch(batch_data)

print(batch_results)


print("\nInference testing completed successfully.")
