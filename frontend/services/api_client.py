
import requests

API_BASE_URL = "http://127.0.0.1:8000"


def check_backend_health():
    response = requests.get(
        f"{API_BASE_URL}/health/ready",
        timeout=10
    )

    response.raise_for_status()
    return response.json()


def predict_single(customer_data: dict):
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=customer_data,
        timeout=30
    )

    response.raise_for_status()
    return response.json()


def predict_batch(file):
    files = {
        "file": (
            file.name,
            file.read(),
            "text/csv"
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/predict/batch",
        files=files,
        timeout=120
    )

    if not response.ok:
        print("API Error:", response.status_code)
        print("Error details:", response.text)

    response.raise_for_status()
    return response.json()


def explain_customer(customer_data: dict):
    response = requests.post(
        f"{API_BASE_URL}/explain",
        json=customer_data,
        timeout=60
    )

    response.raise_for_status()
    return response.json()


def get_explanation_plot(customer_id: str):
    response = requests.get(
        f"{API_BASE_URL}/explain/plot/{customer_id}",
        timeout=30
    )

    response.raise_for_status()
    return response.content



def explain_batch(file):
    """
    Generate batch-level SHAP explanations.
    """
    files = {
        "file": (
            file.name,
            file.getvalue(),
            "text/csv"
        )
    }

    response = requests.post(
        f"{API_BASE_URL}/explain/batch",
        files=files,
        timeout=180
    )

    response.raise_for_status()

    return response.json()


def get_batch_explanation_plot(plot_type: str):
    """
    Retrieve batch-level SHAP plots.

    Supported plot types:
    - beeswarm
    - importance
    """
    if plot_type not in ["beeswarm", "importance"]:
        raise ValueError(
            "Invalid plot type. Use 'beeswarm' or 'importance'."
        )

    response = requests.get(
        f"{API_BASE_URL}/explain/batch/plot/{plot_type}",
        timeout=30
    )

    response.raise_for_status()

    return response.content
