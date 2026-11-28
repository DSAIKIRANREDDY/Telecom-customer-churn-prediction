
import os
import requests
import streamlit as st


def get_config(key: str, default=None):
    """Read configuration from environment variables or Streamlit Secrets."""
    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


API_BASE_URL = get_config(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

API_KEY = get_config("API_KEY")

# Attach authentication to every protected API request
API_HEADERS = (
    {"X-API-Key": API_KEY}
    if API_KEY
    else {}
)


def check_backend_health():
    response = requests.get(
        f"{API_BASE_URL}/health/ready",
        headers=API_HEADERS,
        timeout=10
    )

    response.raise_for_status()
    return response.json()


def predict_single(customer_data: dict):
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=customer_data,
        headers=API_HEADERS,
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
        headers=API_HEADERS,
        timeout=120
    )

    response.raise_for_status()
    return response.json()


def explain_customer(customer_data: dict):
    response = requests.post(
        f"{API_BASE_URL}/explain",
        json=customer_data,
        headers=API_HEADERS,
        timeout=60
    )

    response.raise_for_status()
    return response.json()


def get_explanation_plot(customer_id: str):
    response = requests.get(
        f"{API_BASE_URL}/explain/plot/{customer_id}",
        headers=API_HEADERS,
        timeout=30
    )

    response.raise_for_status()
    return response.content


def explain_batch(file):
    """Generate batch-level SHAP explanations."""

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
        headers=API_HEADERS,
        timeout=180
    )

    response.raise_for_status()
    return response.json()


def get_batch_explanation_plot(plot_type: str):
    """Retrieve batch-level SHAP plots."""

    if plot_type not in ["beeswarm", "importance"]:
        raise ValueError(
            "Invalid plot type. Use 'beeswarm' or 'importance'."
        )

    response = requests.get(
        f"{API_BASE_URL}/explain/batch/plot/{plot_type}",
        headers=API_HEADERS,
        timeout=30
    )

    response.raise_for_status()
    return response.content
