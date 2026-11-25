import streamlit as st
import pandas as pd
import requests

from services.api_client import (
    predict_single,
    explain_customer,
    get_explanation_plot
)

def show_single_prediction():

    st.title("Single Customer Churn Prediction")
    st.caption(
        "Enter customer details to estimate the likelihood "
        "of the customer leaving the telecom company."
    )

    st.divider()

    with st.form("customer_prediction_form"):

        # --------------------------------------------------
        # 1. CUSTOMER PROFILE
        # --------------------------------------------------

        st.subheader("1. Customer Profile")
        st.caption("Basic information about the customer.")

        col1, col2 = st.columns(2)

        with col1:

            customer_id = st.text_input(
                "Customer ID",
                help="A unique identifier assigned to the customer. "
                     "Optional. Example: 7590-VHVEG."
            )

            gender = st.selectbox(
                "Gender",
                ["Female", "Male"],
                help="The gender recorded in the customer's account."
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No",
                help="Whether the customer is classified as a senior "
                     "citizen. 1 = Yes, 0 = No."
            )

        with col2:

            partner = st.selectbox(
                "Partner",
                ["Yes", "No"],
                help="Whether the customer has a partner."
            )

            dependents = st.selectbox(
                "Dependents",
                ["Yes", "No"],
                help="Whether the customer has dependents."
            )

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12,
                help="How many months the customer has been with "
                     "the telecom company. Example: 12 means one year."
            )

        st.divider()

        # --------------------------------------------------
        # 2. SUBSCRIBED SERVICES
        # --------------------------------------------------

        st.subheader("2. Subscribed Services")
        st.caption("Information about the services used by the customer.")

        col1, col2 = st.columns(2)

        with col1:

            phone_service = st.selectbox(
                "Phone Service",
                ["Yes", "No"],
                help="Whether the customer subscribes to phone service."
            )

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"],
                help="Whether the customer has multiple phone lines. "
                     "'No phone service' means the customer does not "
                     "subscribe to phone service."
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"],
                help="The type of internet service used by the customer. "
                     "DSL and Fiber optic are internet service types. "
                     "'No' means no internet service."
            )

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"],
                help="Whether the customer subscribes to an online "
                     "security add-on."
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"],
                help="Whether the customer subscribes to an online "
                     "backup service."
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"],
                help="Whether the customer subscribes to device "
                     "protection services."
            )

        with col2:

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"],
                help="Whether the customer subscribes to technical "
                     "support services."
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"],
                help="Whether the customer subscribes to a streaming "
                     "TV service."
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"],
                help="Whether the customer subscribes to a streaming "
                     "movie service."
            )

        st.divider()

        # --------------------------------------------------
        # 3. BILLING INFORMATION
        # --------------------------------------------------

        st.subheader("3. Billing Information")
        st.caption("Contract, payment and billing details.")

        col1, col2 = st.columns(2)

        with col1:

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
                help="The customer's contract type:\n\n"
                     "- Month-to-month: Monthly agreement.\n"
                     "- One year: One-year agreement.\n"
                     "- Two year: Two-year agreement."
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"],
                help="Whether the customer receives bills "
                     "electronically instead of paper bills."
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                help="The method used by the customer to pay "
                     "their telecom bills."
            )

        with col2:

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0,
                step=1.0,
                help="The amount charged to the customer each month. "
                     "Example: 70 means 70 currency units per month."
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0,
                step=1.0,
                help="The total amount charged to the customer over "
                     "their entire tenure. Example: 840 after 12 months "
                     "with monthly charges of 70."
            )

        st.divider()

        submitted = st.form_submit_button(
            "Predict Customer Churn",
            width="stretch",
            type="primary"
        )

    # --------------------------------------------------
    # 4. GENERATE PREDICTION
    # --------------------------------------------------

    
    if submitted:

        customer_data = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges
        }

        if customer_id.strip():
            customer_data["customerID"] = customer_id.strip()

        # Use an internal identifier if the user leaves Customer ID blank.
        explanation_data = customer_data.copy()

        if "customerID" not in explanation_data:
            explanation_data["customerID"] = "SINGLE-CUSTOMER"

        try:

            with st.spinner(
                "Generating prediction and SHAP explanation..."
            ):

                # 1. Generate churn prediction
                prediction = predict_single(customer_data)

                # 2. Generate individual SHAP explanation
                explanation = explain_customer(explanation_data)

                # 3. Retrieve SHAP waterfall plot
                plot_bytes = get_explanation_plot(
                    explanation["customerID"]
                )

            # Save results in session state
            st.session_state["single_prediction"] = prediction
            st.session_state["single_explanation"] = explanation
            st.session_state["single_explanation_plot"] = plot_bytes

        except requests.exceptions.HTTPError as error:

            st.error(
                "The prediction or explanation request could not "
                "be completed. Please check the customer details."
            )

        except requests.exceptions.RequestException:

            st.error(
                "Unable to connect to the prediction service. "
                "Please check whether FastAPI is running."
            )

        except Exception as error:

            st.error(
                f"An unexpected error occurred: {error}"
            )

    # --------------------------------------------------
    # DISPLAY PREDICTION RESULTS
    # --------------------------------------------------

    if "single_prediction" in st.session_state:

        result = st.session_state["single_prediction"]

        st.divider()
        st.subheader("Prediction Results")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Churn Probability",
            f"{result['churn_probability']:.2%}"
        )

        col2.metric(
            "Predicted Churn",
            result["predicted_churn"]
        )

        col3.metric(
            "Risk Level",
            result["risk_level"]
        )

        st.progress(result["churn_probability"])

        if result["predicted_churn"] == "Yes":

            st.warning(
                "The model predicts that this customer is likely to churn."
            )

        else:

            st.success(
                "The model predicts that this customer is likely to stay."
            )

        # --------------------------------------------------
        # SHAP EXPLANATION
        # --------------------------------------------------

        if "single_explanation" in st.session_state:

            explanation = st.session_state["single_explanation"]

            plot_bytes = st.session_state["single_explanation_plot"]

            st.divider()
            st.subheader("Why Did the Model Make This Prediction?")

            st.caption(
                "The following SHAP waterfall plot explains how "
                "each customer feature contributes to the model's prediction."
            )

            st.image(
                plot_bytes,
                caption="Individual Customer SHAP Waterfall Plot",
                width="stretch"
            )

            st.subheader("Feature Contributions")

            contributions = pd.DataFrame(
                explanation["contributions"]
            )

            if "feature_value" in contributions.columns:
                contributions["feature_value"] = (
                    contributions["feature_value"].astype("string")
                )

            st.caption(
                "Positive SHAP values push the model output towards churn. "
                "Negative SHAP values push it away from churn. "
                "The waterfall plot uses the model's raw output scale, "
                "not probability."
            )

            st.dataframe(
                contributions,
                width="stretch",
                hide_index=True
            )