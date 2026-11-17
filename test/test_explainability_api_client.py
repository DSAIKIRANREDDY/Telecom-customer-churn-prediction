from frontend.services.api_client import (
    explain_customer,
    get_explanation_plot
)

customer_data = {
    "customerID": "TEST-API-001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.7,
    "TotalCharges": 70.7
}

# Request SHAP explanation
explanation = explain_customer(customer_data)

print("Customer ID:", explanation["customerID"])
print("Base value:", explanation["base_value"])
print("\nTop feature contributions:")

for item in explanation["contributions"][:5]:
    print(item)

# Retrieve SHAP waterfall plot
plot = get_explanation_plot("TEST-API-001")

with open("customer_explanation.png", "wb") as file:
    file.write(plot)

print("\nSHAP plot saved successfully.")