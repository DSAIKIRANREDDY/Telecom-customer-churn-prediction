
import pandas as pd
import pytest

from fastapi.testclient import TestClient
from telecom_churn.api.main import app


# --------------------------------------------------
# Test customer data
# --------------------------------------------------

CUSTOMER = {
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


# --------------------------------------------------
# Test client fixture
# --------------------------------------------------

@pytest.fixture(scope="session")
def client():

    # Starts the FastAPI application and runs lifespan
    # Model and SHAP explainer are loaded once per test session.
    with TestClient(app) as test_client:
        yield test_client


# --------------------------------------------------
# 1. Root endpoint test
# --------------------------------------------------

def test_root_endpoint(client):

    response = client.get("/")

    assert response.status_code == 200

    result = response.json()

    assert result["status"] == "running"


# --------------------------------------------------
# 2. Health endpoint test
# --------------------------------------------------

def test_health_endpoint(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert data["components"]["model_loaded"] is True
    assert data["components"]["explainer_loaded"] is True
    assert data["components"]["predictor_loaded"] is True


def test_health_live_endpoint(client):

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json() == {
        "status": "alive"
    }



def test_health_ready_endpoint(client):

    response = client.get("/health/ready")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["components"]["model_loaded"] is True
    assert data["components"]["explainer_loaded"] is True
    assert data["components"]["predictor_loaded"] is True

# --------------------------------------------------
# 3. Single prediction test
# --------------------------------------------------

def test_single_prediction(client):

    response = client.post(
        "/predict",
        json=CUSTOMER
    )

    assert response.status_code == 200

    result = response.json()

    assert result["customerID"] == "TEST-API-001"

    assert 0 <= result["churn_probability"] <= 1

    assert result["predicted_churn"] in ["Yes", "No"]

    assert result["risk_level"] in [
        "Low",
        "Medium",
        "High"
    ]


# --------------------------------------------------
# 4. Invalid single prediction test
# --------------------------------------------------

def test_invalid_customer_request(client):

    invalid_customer = CUSTOMER.copy()

    # Remove a required feature
    invalid_customer.pop("tenure")

    response = client.post(
        "/predict",
        json=invalid_customer
    )

    assert response.status_code == 422


# --------------------------------------------------
# 5. Batch prediction test
# --------------------------------------------------

def test_batch_prediction(client):

    batch_df = pd.DataFrame([
        CUSTOMER,
        {
            **CUSTOMER,
            "customerID": "TEST-API-002"
        },
        {
            **CUSTOMER,
            "customerID": "TEST-API-003"
        }
    ])

    csv_content = batch_df.to_csv(
        index=False
    ).encode("utf-8")

    response = client.post(
        "/predict/batch",
        files={
            "file": (
                "test_customers.csv",
                csv_content,
                "text/csv"
            )
        }
    )

    assert response.status_code == 200

    result = response.json()

    assert result["total_customers"] == 3

    assert len(result["predictions"]) == 3

    for prediction in result["predictions"]:

        assert 0 <= prediction["churn_probability"] <= 1

        assert prediction["predicted_churn"] in ["Yes", "No"]


# --------------------------------------------------
# 6. Invalid batch file test
# --------------------------------------------------

def test_invalid_batch_file(client):

    response = client.post(
        "/predict/batch",
        files={
            "file": (
                "customers.txt",
                b"some invalid content",
                "text/plain"
            )
        }
    )

    assert response.status_code == 400


# --------------------------------------------------
# 7. Customer explanation test
# --------------------------------------------------

def test_customer_explanation(client):

    response = client.post(
        "/explain",
        json=CUSTOMER
    )

    assert response.status_code == 200

    result = response.json()

    assert result["customerID"] == "TEST-API-001"

    assert "base_value" in result

    assert len(result["contributions"]) > 0

    assert result["plot_url"] == (
        "/explain/plot/TEST-API-001"
    )


# --------------------------------------------------
# 8. Waterfall plot retrieval test
# --------------------------------------------------

def test_customer_explanation_plot(client):

    # Generate the explanation first
    explanation_response = client.post(
        "/explain",
        json=CUSTOMER
    )

    assert explanation_response.status_code == 200

    plot_url = explanation_response.json()["plot_url"]

    # Retrieve generated image
    response = client.get(plot_url)

    assert response.status_code == 200

    assert response.headers["content-type"] == "image/png"

    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"



def test_invalid_categorical_value(client):
    customer_data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "InvalidContract",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.5,
        "TotalCharges": 850.0
    }

    response = client.post("/predict", json=customer_data)

    assert response.status_code == 422



def test_batch_prediction_invalid_categorical_value(client):

    csv_content = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
TEST-001,Male,0,Yes,No,12,Yes,No,DSL,No,Yes,No,No,Yes,No,InvalidContract,Yes,Electronic check,70.5,850.0
"""

    files = {
        "file": (
            "invalid_customer.csv",
            csv_content,
            "text/csv"
        )
    }

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 422

    response_data = response.json()

    assert response_data["detail"]["message"] == (
        "CSV contains invalid customer data"
    )

    assert response_data["detail"]["errors"][0]["field"] == "Contract"



def test_batch_prediction_missing_column(client):

    csv_content = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
TEST-001,Male,0,Yes,No,12,Yes,No,DSL,No,Yes,No,No,Yes,No,Yes,Electronic check,70.5,850.0
"""

    files = {
        "file": (
            "missing_column.csv",
            csv_content,
            "text/csv"
        )
    }

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 422

    response_data = response.json()

    assert "Missing required columns" in response_data["detail"]
    assert "Contract" in response_data["detail"]



def test_batch_prediction_unexpected_column(client):

    csv_content = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,CustomerAge
TEST-001,Male,0,Yes,No,12,Yes,No,DSL,No,Yes,No,No,Yes,No,Month-to-month,Yes,Electronic check,70.5,850.0,35
"""

    files = {
        "file": (
            "unexpected_column.csv",
            csv_content,
            "text/csv"
        )
    }

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 422

    response_data = response.json()

    assert "Unexpected columns" in response_data["detail"]
    assert "CustomerAge" in response_data["detail"]



def test_batch_prediction_exceeds_maximum_limit(client):

    from telecom_churn.api.main import MAX_BATCH_SIZE

    # Generate more records than the allowed limit
    customer_data = {
        "customerID": ["TEST-001"] * (MAX_BATCH_SIZE + 1),
        "gender": ["Male"] * (MAX_BATCH_SIZE + 1),
        "SeniorCitizen": [0] * (MAX_BATCH_SIZE + 1),
        "Partner": ["Yes"] * (MAX_BATCH_SIZE + 1),
        "Dependents": ["No"] * (MAX_BATCH_SIZE + 1),
        "tenure": [12] * (MAX_BATCH_SIZE + 1),
        "PhoneService": ["Yes"] * (MAX_BATCH_SIZE + 1),
        "MultipleLines": ["No"] * (MAX_BATCH_SIZE + 1),
        "InternetService": ["DSL"] * (MAX_BATCH_SIZE + 1),
        "OnlineSecurity": ["No"] * (MAX_BATCH_SIZE + 1),
        "OnlineBackup": ["Yes"] * (MAX_BATCH_SIZE + 1),
        "DeviceProtection": ["No"] * (MAX_BATCH_SIZE + 1),
        "TechSupport": ["No"] * (MAX_BATCH_SIZE + 1),
        "StreamingTV": ["Yes"] * (MAX_BATCH_SIZE + 1),
        "StreamingMovies": ["No"] * (MAX_BATCH_SIZE + 1),
        "Contract": ["Month-to-month"] * (MAX_BATCH_SIZE + 1),
        "PaperlessBilling": ["Yes"] * (MAX_BATCH_SIZE + 1),
        "PaymentMethod": ["Electronic check"] * (MAX_BATCH_SIZE + 1),
        "MonthlyCharges": [70.5] * (MAX_BATCH_SIZE + 1),
        "TotalCharges": [850.0] * (MAX_BATCH_SIZE + 1)
    }

    customers_df = pd.DataFrame(customer_data)

    csv_content = customers_df.to_csv(index=False)

    files = {
        "file": (
            "large_batch.csv",
            csv_content,
            "text/csv"
        )
    }

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 413

    assert response.json()["detail"] == (
        f"Batch size exceeds the maximum limit of {MAX_BATCH_SIZE} customers"
    )



def test_batch_prediction_exceeds_file_size_limit(client):

    from telecom_churn.api.main import MAX_UPLOAD_SIZE

    oversized_content = b"A" * (MAX_UPLOAD_SIZE + 1)

    files = {
        "file": (
            "large_file.csv",
            oversized_content,
            "text/csv"
        )
    }

    response = client.post("/predict/batch", files=files)

    assert response.status_code == 413

    assert response.json()["detail"] == (
        "Uploaded file exceeds the maximum allowed size of 10 MB"
    )