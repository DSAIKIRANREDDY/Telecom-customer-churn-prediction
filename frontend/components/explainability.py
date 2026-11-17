
import streamlit as st
import pandas as pd
import requests

from services.api_client import (
    explain_customer,
    get_explanation_plot
)


def show_explainability():

    st.title("Customer Churn Explainability")

    st.caption(
        "Understand why the machine learning model predicts "
        "a particular customer to churn or stay."
    )

    st.divider()

    st.subheader("Enter Customer Information")

    st.info(
        "Enter the customer's details below. The SHAP explainer "
        "will identify which features contribute to the prediction."
    )

    with st.form("customer_explainability_form"):

        customer_id = st.text_input(
            "Customer ID",
            value="CUSTOMER-EXPLAIN-001",
            help="Enter a unique customer identifier for this explanation."
        )

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox("Gender", ["Female", "Male"])

            senior_citizen = st.selectbox(
                "Senior Citizen",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            partner = st.selectbox("Partner", ["Yes", "No"])

            dependents = st.selectbox("Dependents", ["Yes", "No"])

            tenure = st.number_input(
                "Tenure (Months)",
                min_value=0,
                max_value=100,
                value=12
            )

            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )

            internet_service = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )

            contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"]
            )

        with col2:

            online_security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )

            online_backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

            device_protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )

            tech_support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )

            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )

            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

            paperless_billing = st.selectbox(
                "Paperless Billing",
                ["Yes", "No"]
            )

            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

            monthly_charges = st.number_input(
                "Monthly Charges",
                min_value=0.0,
                value=70.0
            )

            total_charges = st.number_input(
                "Total Charges",
                min_value=0.0,
                value=840.0
            )

        submitted = st.form_submit_button(
            "Explain Customer Churn",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        if not customer_id.strip():
            st.error("Please enter a Customer ID.")
            return

        customer_data = {
            "customerID": customer_id.strip(),
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

        try:

            with st.spinner("Generating SHAP explanation..."):

                result = explain_customer(customer_data)

                plot_bytes = get_explanation_plot(
                    result["customerID"]
                )

            st.session_state["customer_explanation"] = result
            st.session_state["customer_explanation_plot"] = plot_bytes

        except requests.exceptions.HTTPError as error:

            if error.response is not None:
                st.error(
                    "Unable to generate the explanation. "
                    "Please check the customer information and try again."
                )
            else:
                st.error("Unable to generate the customer explanation.")

        except requests.exceptions.RequestException:
            st.error(
                "Unable to connect to the prediction service. "
                "Please check whether FastAPI is running."
            )

        except Exception as error:
            st.error(f"An unexpected error occurred: {error}")

    # --------------------------------------------------
    # DISPLAY EXPLANATION
    # --------------------------------------------------

    if "customer_explanation" in st.session_state:

        result = st.session_state["customer_explanation"]

        plot_bytes = st.session_state["customer_explanation_plot"]

        st.divider()
        st.subheader("SHAP Waterfall Explanation")

        st.caption(
            "The waterfall plot shows how individual customer features "
            "move the model output away from its baseline prediction."
        )

        st.image(
            plot_bytes,
            caption=f"SHAP explanation for {result['customerID']}",
            use_container_width=True
        )

        st.divider()
        st.subheader("Feature Contributions")

        contributions = pd.DataFrame(result["contributions"])

        if not contributions.empty:

            display_df = contributions[
                ["feature", "feature_value", "shap_value", "impact"]
            ].copy()

            display_df["shap_value"] = display_df["shap_value"].round(4)

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Positive SHAP values push the model output towards churn. "
                "Negative SHAP values push it away from churn."
            )
