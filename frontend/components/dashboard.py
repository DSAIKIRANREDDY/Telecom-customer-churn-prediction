
import streamlit as st
import requests

from services.api_client import check_backend_health


def show_dashboard():

    # Page heading
    st.title("Telecom Churn Dashboard")

    st.caption(
        "Overview of customer churn data and machine learning model"
    )

    st.divider()

    # ---------------------------------------------
    # Customer Statistics
    # ---------------------------------------------

    st.subheader("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        label="Total Customers",
        value="7,043"
    )

    col2.metric(
        label="Churned Customers",
        value="1,869"
    )

    col3.metric(
        label="Retained Customers",
        value="5,174"
    )

    col4.metric(
        label="Churn Rate",
        value="26.54%"
    )

    st.divider()

    # ---------------------------------------------
    # Model Information
    # ---------------------------------------------

    st.subheader("Model Information")

    col1, col2, col3 = st.columns(3)

    col1.info(
        "**Algorithm**\n\nTuned XGBoost"
    )

    col2.info(
        "**Prediction Threshold**\n\n0.65"
    )

    col3.info(
        "**Explainability**\n\nSHAP"
    )

    st.divider()

    # ---------------------------------------------
    # Backend Connection
    # ---------------------------------------------

    st.subheader("Backend Status")

    try:

        health = check_backend_health()

        if health["status"] == "ready":

            st.success("Backend is connected and ready.")

            components = health["components"]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Model Loaded",
                "Yes" if components["model_loaded"] else "No"
            )

            col2.metric(
                "SHAP Explainer",
                "Ready" if components["explainer_loaded"] else "Not Ready"
            )

            col3.metric(
                "Predictor",
                "Ready" if components["predictor_loaded"] else "Not Ready"
            )

        else:
            st.error("Backend is not ready.")

            st.json(health)

    except requests.exceptions.RequestException:

        st.error(
            "Unable to connect to FastAPI. "
            "Please make sure the backend is running."
        )
