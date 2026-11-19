
import streamlit as st


def show_about():

    st.title("About the Project")

    st.markdown(
        """
        ## Telecom Customer Churn Prediction

        This project uses Machine Learning to predict whether a
        telecom customer is likely to discontinue their subscription.

        The application allows users to evaluate individual customers
        or upload a CSV file to generate predictions for multiple
        customers.
        """
    )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown(
        """
        - Predict customer churn using a trained Machine Learning model.
        - Identify customers with different levels of churn risk.
        - Explain individual predictions using SHAP.
        - Support batch predictions through CSV uploads.
        - Provide an easy-to-use interface for interacting with the model.
        """
    )

    st.divider()

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Customers", "7,043")

    with col2:
        st.metric("Customer Features", "19")

    with col3:
        st.metric("Target Variable", "Churn")

    st.markdown(
        """
        **Target classes:**
        - Yes — Customer churned.
        - No — Customer retained the subscription.
        """
    )

    st.divider()

    st.subheader("Machine Learning Approach")

    st.markdown(
        """
        1. Data ingestion and validation
        2. Data preprocessing and feature engineering
        3. Model training and comparison
        4. Hyperparameter optimization using Optuna
        5. Classification threshold optimization
        6. Model evaluation
        7. Model interpretability and SHAP explainability
        """
    )

    st.markdown(
        """
        **Selected model:** XGBoost Classifier

        **Classification threshold:** 0.65
        """
    )

    st.divider()

    st.subheader("Technology Stack")

    st.markdown(
        """
        | Area | Technologies |
        |---|---|
        | Programming | Python |
        | Data processing | Pandas, NumPy |
        | Machine Learning | Scikit-learn, XGBoost |
        | Hyperparameter tuning | Optuna |
        | Model explainability | SHAP |
        | Backend | FastAPI |
        | Frontend | Streamlit |
        | API testing | Pytest, HTTPX |
        | Dependency management | uv |
        """
    )

    st.divider()

    st.subheader("Source Code")

    st.markdown(
        """
        The complete project source code is available on GitHub.
        """
    )

    st.markdown(
        "[View GitHub Repository](https://github.com/DSAIKIRANREDDY/Telecom-customer-churn-prediction.git)"
    )

    st.divider()

    st.caption(
        "Telecom Customer Churn Prediction | Machine Learning Project"
    )
