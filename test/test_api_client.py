from frontend.services.api_client import (
    check_backend_health,
    predict_single
)

# Check backend readiness
health = check_backend_health()

print("Backend health:")
print(health)


# Use the valid customer_data dictionary
# from your existing test_single_prediction test
customer_data = {
    "customerID": "TEST-API-001",
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 1026.0
}

# Generate prediction through Streamlit API client
result = predict_single(customer_data)

print("\nPrediction result:")
print(result)