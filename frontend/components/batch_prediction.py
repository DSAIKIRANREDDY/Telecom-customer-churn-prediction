
import streamlit as st
import pandas as pd
import requests

from services.api_client import (
    predict_batch,
    explain_customer,
    get_explanation_plot,
    explain_batch,
    get_batch_explanation_plot
)


class UploadedCSV:
    """Small wrapper for reusing uploaded CSV bytes."""

    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


def display_batch_error(error):

    response = error.response

    if response is None:
        st.error(
            "We couldn't connect to the prediction service. "
            "Please try again later."
        )
        return

    try:
        error_data = response.json()
    except ValueError:
        st.error(
            "Something went wrong while processing your file. "
            "Please check the CSV and try again."
        )
        return

    detail = error_data.get("detail", {})

    # FastAPI validation errors for individual CSV rows
    if isinstance(detail, dict) and "errors" in detail:

        st.error(
            "Some customer records contain invalid information. "
            "Please correct the details below and upload the file again."
        )

        for item in detail["errors"]:

            row = item.get("row", "Unknown")
            field = item.get("field", "Unknown field")
            message = item.get("message", "Invalid value")

            with st.container(border=True):

                st.markdown(f"**Row {row} — {field}**")
                st.warning(message)

                if field == "Contract":
                    st.info(
                        "Valid values: Month-to-month, One year, Two year"
                    )

                elif field == "gender":
                    st.info("Valid values: Male, Female")

                elif field == "SeniorCitizen":
                    st.info("Valid values: 0 for No or 1 for Yes")

                else:
                    st.info(
                        f"Please correct the value in the '{field}' column."
                    )

        return

    # Other errors returned by the backend
    if isinstance(detail, dict):
        message = detail.get("message", "Unable to process the file.")
    else:
        message = str(detail)

    status_code = response.status_code

    if status_code == 413:
        st.error(
            "Your file is too large or contains too many customer records. "
            "Please upload a smaller file."
        )

    elif status_code == 422:
        st.error(
            f"We couldn't validate your CSV file. {message}"
        )

    elif status_code == 400:
        st.error(
            f"Your CSV file could not be processed. {message}"
        )

    else:
        st.error(
            "An unexpected error occurred while processing your request. "
            "Please try again later."
        )


def show_batch_prediction():

    st.title("Batch Customer Churn Prediction")

    st.caption(
        "Upload a CSV file containing multiple customer records "
        "to generate churn predictions."
    )

    st.divider()

    st.subheader("Upload Customer Data")

    st.info(
        """
        **Before uploading:**

        - Upload a CSV file.
        - Include all 19 required customer features.
        - You may optionally include `customerID`.
        - Maximum file size: 10 MB.
        - Maximum number of customers: 5,000.
        """
    )

    uploaded_file = st.file_uploader(
        "Choose a customer CSV file",
        type=["csv"],
        help="Upload a CSV containing customer details for prediction."
    )

    if uploaded_file is not None:

        st.divider()
        st.subheader("Uploaded File Preview")

        try:
            uploaded_file.seek(0)
            preview_df = pd.read_csv(uploaded_file)

            if preview_df.empty:
                st.warning("The uploaded CSV contains no customer records.")
                return

            st.write(f"**Number of records:** {len(preview_df)}")
            st.write(f"**Number of columns:** {len(preview_df.columns)}")

            st.dataframe(
                preview_df.head(10),
                use_container_width=True
            )

            if len(preview_df) > 5000:
                st.error(
                    "The file contains more than 5,000 records. "
                    "Please upload a smaller file."
                )
                return

            if len(preview_df.columns) != len(set(preview_df.columns)):
                st.error("The uploaded file contains duplicate column names.")
                return

        except pd.errors.EmptyDataError:
            st.error("The uploaded CSV file is empty.")
            return

        except (pd.errors.ParserError, UnicodeDecodeError):
            st.error("Unable to read the CSV file. Please check its format.")
            return

        except Exception as error:
            st.error(f"Unable to preview the uploaded file: {error}")
            return

        st.divider()

        if st.button(
            "Generate Batch Predictions",
            type="primary",
            use_container_width=True
        ):

            try:
                uploaded_file.seek(0)

                # Preserve original CSV bytes for later SHAP analysis.
                csv_bytes = uploaded_file.getvalue()
                csv_name = uploaded_file.name

                with st.spinner(
                    f"Generating predictions for {len(preview_df)} customers..."
                ):
                    response = predict_batch(uploaded_file)

                if isinstance(response, dict):
                    predictions = response.get(
                        "predictions",
                        response.get("results", [])
                    )
                else:
                    predictions = response

                if not predictions:
                    st.warning("No prediction results were returned.")
                    return

                results_df = pd.DataFrame(predictions)

                st.session_state["batch_prediction_results"] = results_df
                st.session_state["batch_customer_data"] = preview_df.copy()

                # Save the CSV so the same batch can be explained later.
                st.session_state["batch_upload_bytes"] = csv_bytes
                st.session_state["batch_upload_name"] = csv_name

                # Clear explanations belonging to an older batch.
                for key in [
                    "batch_shap_result",
                    "batch_beeswarm_plot",
                    "batch_importance_plot",
                    "batch_selected_explanation",
                    "batch_selected_plot"
                ]:
                    st.session_state.pop(key, None)

                st.success(
                    f"Successfully generated predictions for "
                    f"{len(results_df)} customers!"
                )

            except requests.exceptions.HTTPError as error:
                display_batch_error(error)

            except requests.exceptions.RequestException as error:
                st.error(f"Unable to connect to FastAPI: {error}")

            except Exception as error:
                st.error(f"An unexpected error occurred: {error}")

    # --------------------------------------------------
    # DISPLAY PREDICTION RESULTS
    # --------------------------------------------------

    if "batch_prediction_results" in st.session_state:

        results_df = st.session_state["batch_prediction_results"]

        st.divider()
        st.subheader("Batch Prediction Results")

        col1, col2, col3 = st.columns(3)

        total_customers = len(results_df)

        churned = (
            results_df["predicted_churn"] == "Yes"
        ).sum()

        retained = (
            results_df["predicted_churn"] == "No"
        ).sum()

        col1.metric("Total Customers", total_customers)
        col2.metric("Predicted Churn", int(churned))
        col3.metric("Predicted Retained", int(retained))

        st.dataframe(
            results_df.style.format(
                {"churn_probability": "{:.2%}"}
            ),
            use_container_width=True
        )

        csv_data = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="customer_churn_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )

        # --------------------------------------------------
        # BATCH-LEVEL SHAP EXPLAINABILITY
        # --------------------------------------------------

        st.divider()
        st.subheader("Batch-Level SHAP Analysis")

        st.caption(
            "Understand which customer features influence churn predictions "
            "across the entire uploaded batch."
        )

        if "batch_upload_bytes" in st.session_state:

            if st.button(
                "Generate Batch SHAP Analysis",
                use_container_width=True
            ):

                try:
                    # Reconstruct the original CSV upload.
                    csv_file = UploadedCSV(
                        name=st.session_state["batch_upload_name"],
                        content=st.session_state["batch_upload_bytes"]
                    )

                    with st.spinner(
                        "Generating SHAP explanations for all customers..."
                    ):

                        shap_result = explain_batch(csv_file)

                        beeswarm_plot = get_batch_explanation_plot(
                            "beeswarm"
                        )

                        importance_plot = get_batch_explanation_plot(
                            "importance"
                        )

                    st.session_state["batch_shap_result"] = shap_result
                    st.session_state["batch_beeswarm_plot"] = beeswarm_plot
                    st.session_state["batch_importance_plot"] = importance_plot

                    st.success(
                        "Batch-level SHAP analysis generated successfully!"
                    )

                except requests.exceptions.HTTPError as error:
                    display_batch_error(error)

                except requests.exceptions.RequestException:
                    st.error(
                        "Unable to connect to the explanation service."
                    )

                except Exception as error:
                    st.error(
                        f"Unable to generate batch SHAP analysis: {error}"
                    )

        else:
            st.info(
                "Generate batch predictions first to enable batch SHAP analysis."
            )

        # --------------------------------------------------
        # DISPLAY BATCH SHAP RESULTS
        # --------------------------------------------------

        if "batch_shap_result" in st.session_state:

            shap_result = st.session_state["batch_shap_result"]

            st.divider()

            st.markdown("#### Batch Explanation Overview")

            st.metric(
                "Customers Explained",
                shap_result["total_customers"]
            )

            # SHAP Beeswarm plot
            st.markdown("#### SHAP Beeswarm Plot")

            st.caption(
                "Shows how each feature influences churn predictions "
                "across the uploaded customers. Each point represents "
                "a customer's feature contribution."
            )

            st.image(
                st.session_state["batch_beeswarm_plot"],
                use_container_width=True
            )

            # SHAP Feature Importance plot
            st.markdown("#### Batch Feature Importance")

            st.caption(
                "Features are ranked using their mean absolute SHAP values "
                "across the uploaded customers."
            )

            st.image(
                st.session_state["batch_importance_plot"],
                use_container_width=True
            )

            # Feature importance table
            st.markdown("#### Feature Importance Values")

            importance_df = pd.DataFrame(
                shap_result["feature_importance"]
            )

            if not importance_df.empty:

                importance_df = importance_df.rename(
                    columns={
                        "feature": "Feature",
                        "mean_absolute_shap": "Mean Absolute SHAP"
                    }
                )

                importance_df["Mean Absolute SHAP"] = (
                    importance_df["Mean Absolute SHAP"].round(5)
                )

                st.dataframe(
                    importance_df,
                    use_container_width=True,
                    hide_index=True
                )

            st.caption(
                "Higher mean absolute SHAP values indicate a greater "
                "average contribution to the model output. They do not "
                "indicate whether a feature increases or decreases churn."
            )

        # --------------------------------------------------
        # INDIVIDUAL CUSTOMER EXPLAINABILITY
        # --------------------------------------------------

        st.divider()
        st.subheader("Individual Customer Explainability")

        st.caption(
            "Select a customer from the batch prediction results "
            "to understand why the model made that prediction."
        )

        if "batch_customer_data" in st.session_state:

            customer_data_df = st.session_state["batch_customer_data"]

            if "customerID" not in customer_data_df.columns:
                st.warning(
                    "Customer IDs are required to select individual "
                    "customers for explainability."
                )

            else:

                customer_ids = (
                    customer_data_df["customerID"].astype(str).tolist()
                )

                selected_customer_id = st.selectbox(
                    "Select Customer",
                    customer_ids,
                    help="Choose a customer to view their individual SHAP explanation."
                )

                if st.button(
                    "Explain Selected Customer",
                    use_container_width=True
                ):

                    try:

                        selected_row = customer_data_df[
                            customer_data_df["customerID"].astype(str)
                            == selected_customer_id
                        ].iloc[0]

                        customer_data = selected_row.to_dict()

                        with st.spinner(
                            "Generating individual customer explanation..."
                        ):

                            explanation = explain_customer(customer_data)

                            plot_bytes = get_explanation_plot(
                                explanation["customerID"]
                            )

                        st.session_state["batch_selected_explanation"] = explanation
                        st.session_state["batch_selected_plot"] = plot_bytes

                        st.success(
                            f"Explanation generated for {selected_customer_id}"
                        )

                    except requests.exceptions.HTTPError:

                        st.error(
                            "Unable to generate the explanation. "
                            "Please verify the selected customer data."
                        )

                    except requests.exceptions.RequestException:

                        st.error(
                            "Unable to connect to the prediction service."
                        )

                    except Exception as error:

                        st.error(
                            f"An unexpected error occurred: {error}"
                        )

        # --------------------------------------------------
        # DISPLAY INDIVIDUAL SHAP EXPLANATION
        # --------------------------------------------------

        if "batch_selected_explanation" in st.session_state:

            explanation = st.session_state["batch_selected_explanation"]

            st.divider()
            st.subheader(
                f"SHAP Explanation: {explanation['customerID']}"
            )

            st.image(
                st.session_state["batch_selected_plot"],
                caption="Individual Customer SHAP Waterfall Plot",
                use_container_width=True
            )

            st.subheader("Feature Contributions")

            contributions = pd.DataFrame(
                explanation["contributions"]
            )

            # Convert mixed feature values into a consistent string type.
            if "feature_value" in contributions.columns:
                contributions["feature_value"] = (
                    contributions["feature_value"].astype("string")
                )

            st.dataframe(
                contributions,
                width="stretch",
                hide_index=True
            )

            st.caption(
                "Positive SHAP values push the model output towards churn. "
                "Negative SHAP values push it away from churn."
            )
