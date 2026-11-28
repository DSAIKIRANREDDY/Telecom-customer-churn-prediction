
import os
import base64
import requests
import streamlit as st


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

GITHUB_OWNER = "DSAIKIRANREDDY"
GITHUB_REPO = "telecom-churn-config"
GITHUB_BRANCH = "main"
GITHUB_FILE_PATH = "config.json"

GITHUB_CONFIG_URL = (
    f"https://api.github.com/repos/"
    f"{GITHUB_OWNER}/{GITHUB_REPO}/contents/{GITHUB_FILE_PATH}"
)


def get_config(key: str, default=None):
    """Read configuration from environment variables or Streamlit Secrets."""

    value = os.getenv(key)

    if value:
        return value

    try:
        return st.secrets.get(key, default)
    except Exception:
        return default


API_KEY = get_config("API_KEY")

GITHUB_CONFIG_READ_TOKEN = get_config(
    "GITHUB_CONFIG_READ_TOKEN"
)

API_HEADERS = (
    {"X-API-Key": API_KEY}
    if API_KEY
    else {}
)


# ---------------------------------------------------------
# GitHub Configuration Reader
# ---------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def fetch_github_config():
    """
    Fetch the latest configuration from the private GitHub repository.

    The result is cached for 30 seconds.
    """

    if not GITHUB_CONFIG_READ_TOKEN:
        raise RuntimeError(
            "GITHUB_CONFIG_READ_TOKEN is not configured."
        )

    headers = {
        "Authorization": f"Bearer {GITHUB_CONFIG_READ_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "telecom-churn-streamlit",
    }

    response = requests.get(
        GITHUB_CONFIG_URL,
        headers=headers,
        params={"ref": GITHUB_BRANCH},
        timeout=15,
    )

    response.raise_for_status()

    file_data = response.json()

    encoded_content = file_data["content"]

    decoded_content = base64.b64decode(
        encoded_content
    ).decode("utf-8")

    import json

    config = json.loads(decoded_content)

    return config


# ---------------------------------------------------------
# Dynamic API Base URL
# ---------------------------------------------------------

def get_api_base_url():
    """
    Retrieve the current active API URL from GitHub.

    For local development, fall back to localhost
    when the GitHub read token is not configured.
    """

    if not GITHUB_CONFIG_READ_TOKEN:

        local_url = get_config(
            "API_BASE_URL",
            "http://127.0.0.1:8000"
        )

        return local_url.rstrip("/")

    config = fetch_github_config()

    api_base_url = config.get("api_base_url")
    status = config.get("status")

    if status != "active":
        raise RuntimeError(
            f"Backend tunnel is not active. Current status: {status}"
        )

    if not api_base_url:
        raise RuntimeError(
            "API base URL is missing from GitHub configuration."
        )

    if not api_base_url.startswith("https://"):
        raise RuntimeError(
            "Invalid API URL. HTTPS is required."
        )

    return api_base_url.rstrip("/")


# ---------------------------------------------------------
# API Request Helper
# ---------------------------------------------------------

def api_request(method: str, endpoint: str, timeout: int = 30, **kwargs):
    """Send an authenticated request to the current FastAPI URL."""

    api_base_url = get_api_base_url()

    response = requests.request(
        method=method,
        url=f"{api_base_url}{endpoint}",
        headers={
            **API_HEADERS,
            **kwargs.pop("headers", {}),
        },
        timeout=timeout,
        **kwargs,
    )

    response.raise_for_status()

    return response


# ---------------------------------------------------------
# Backend Health Check
# ---------------------------------------------------------

def check_backend_health():

    response = api_request(
        "GET",
        "/health/ready",
        timeout=10,
    )

    return response.json()


# ---------------------------------------------------------
# Single Prediction
# ---------------------------------------------------------

def predict_single(customer_data: dict):

    response = api_request(
        "POST",
        "/predict",
        json=customer_data,
        timeout=30,
    )

    return response.json()


# ---------------------------------------------------------
# Batch Prediction
# ---------------------------------------------------------

def predict_batch(file):

    files = {
        "file": (
            file.name,
            file.getvalue(),
            "text/csv"
        )
    }

    response = api_request(
        "POST",
        "/predict/batch",
        files=files,
        timeout=120,
    )

    return response.json()


# ---------------------------------------------------------
# Single Customer SHAP Explanation
# ---------------------------------------------------------

def explain_customer(customer_data: dict):

    response = api_request(
        "POST",
        "/explain",
        json=customer_data,
        timeout=60,
    )

    return response.json()


def get_explanation_plot(customer_id: str):

    response = api_request(
        "GET",
        f"/explain/plot/{customer_id}",
        timeout=30,
    )

    return response.content


# ---------------------------------------------------------
# Batch SHAP Explanation
# ---------------------------------------------------------

def explain_batch(file):
    """Generate batch-level SHAP explanations."""

    files = {
        "file": (
            file.name,
            file.getvalue(),
            "text/csv"
        )
    }

    response = api_request(
        "POST",
        "/explain/batch",
        files=files,
        timeout=180,
    )

    return response.json()


def get_batch_explanation_plot(plot_type: str):
    """Retrieve batch-level SHAP plots."""

    if plot_type not in ["beeswarm", "importance"]:
        raise ValueError(
            "Invalid plot type. Use 'beeswarm' or 'importance'."
        )

    response = api_request(
        "GET",
        f"/explain/batch/plot/{plot_type}",
        timeout=30,
    )

    return response.content
