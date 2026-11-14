
import pandas as pd
import numpy as np
import joblib
import yaml
import shap
import matplotlib.pyplot as plt

from telecom_churn.entity.config_entity import ModelExplainabilityConfig
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class ModelExplainability:

    def __init__(self, config: ModelExplainabilityConfig):
        self.config = config

    def initiate_model_explainability(self):

        try:
            logger.info("Model explainability started")

            # Load validation dataset
            logger.info("Loading validation dataset")

            validation_df = pd.read_csv(
                self.config.validation_data_path
            )

            # Load tuned model
            logger.info("Loading truned model")

            model = joblib.load(self.config.model_path)

            # Load schema
            logger.info("Loading schema")

            with open(self.config.schema_path, "r", encoding="utf-8") as file:
                schema = yaml.safe_load(file)

            # Separate input features and target
            target_column = schema["target_column"]
            identifier_columns = schema["identifier_columns"]

            X = validation_df.drop(
                columns=[target_column] + identifier_columns
            )

            y = validation_df[target_column].map({
                "No": 0,
                "Yes": 1
            })

            if y.isna().any():
                raise ValueError("Invalid target values found")

            logger.info("Validation data loaded successfully")

            # Extract preprocessor and classifier
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            # Select reproducible sample
            sample_size = min(self.config.sample_size, len(X))

            sampled_rows = validation_df.sample(
                n=sample_size,
                random_state=self.config.random_state
            )

            X_sample = sampled_rows.drop(
                columns=[target_column] + identifier_columns
            ).reset_index(drop=True)

            customer_ids_sample = sampled_rows["customerID"].reset_index(drop=True)

            logger.info("Selected SHAP sample: %s", X_sample.shape)

            # Transform sample using fitted preprocessor
            X_transformed = preprocessor.transform(X_sample)

            feature_names = preprocessor.get_feature_names_out()

            logger.info(
                "Transformed SHAP input shape: %s",
                X_transformed.shape
            )
            explainer, shap_values = self.calculate_shap_values( classifier, X_transformed)
            self.generate_shap_summary_plot(
                shap_values,
                X_transformed,
                feature_names
            )

            for customer_index in self.config.local_explanation_indices:
                self.generate_local_explanation(
                    explainer,
                    shap_values,
                    X_transformed,
                    customer_ids_sample,
                    X_sample,
                    feature_names,
                    customer_index=customer_index
                )

            self.save_shap_values(shap_values,feature_names,customer_ids_sample)
            customer_level_report = self.generate_customer_level_shap_report(
                shap_values=shap_values,
                feature_names=feature_names,
                X_transformed=X_transformed,
                customer_ids=customer_ids_sample
            )

            # Aggregate transformed SHAP values into original features
            original_feature_shap_report = self.aggregate_shap_values(
                shap_values=shap_values,
                feature_names=feature_names,
                X_sample=X_sample,
                customer_ids=customer_ids_sample,
                numerical_features=schema["numerical_columns"],
                categorical_features=[
                    column
                    for column in schema["categorical_columns"]
                    if column != target_column
                ]
            )
            # Save original feature-level SHAP report
            output_path = (
                self.config.report_output_dir
                / self.config.original_feature_shap_report_file
            )
            original_feature_shap_report.to_csv(
                output_path,
                index=False
            )
            logger.info(
                "Original feature-level SHAP report saved: %s",
                output_path
            )

            return {
                "model": model,
                "classifier": classifier,
                "preprocessor": preprocessor,
                "X_sample": X_sample,
                "X_transformed": X_transformed,
                "feature_names": feature_names,
                "explainer": explainer,
                "shap_values": shap_values,
                "customer_ids_sample": customer_ids_sample,
                "original_feature_shap_report": original_feature_shap_report
            }

        except Exception:
            logger.exception("Model explainability failed")
            raise



    def calculate_shap_values(self, classifier, X_transformed):

        try:
            logger.info("Initializing SHAP TreeExplainer")

            # Create TreeExplainer
            explainer = shap.TreeExplainer(classifier)

            logger.info("Calculating SHAP values")

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_transformed)

            logger.info(
                "SHAP values calculated successfully. Shape: %s",
                shap_values.shape
            )

            return explainer, shap_values

        except Exception:
            logger.exception("SHAP value calculation failed")
            raise



    def generate_shap_summary_plot( self, shap_values, X_transformed, feature_names):

        try:
            logger.info("Generating SHAP summary plot")

            plt.figure(figsize=(12, 8))

            shap.summary_plot(
                shap_values,
                X_transformed,
                feature_names=feature_names,
                max_display=self.config.top_n_features,
                show=False
            )

            plt.title("SHAP Global Feature Importance - Tuned XGBoost")
            plt.tight_layout()

            plot_path = (
                self.config.report_output_dir
                / self.config.shap_summary_plot
            )

            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            logger.info("SHAP summary plot saved: %s", plot_path)

        except Exception:
            logger.exception("SHAP summary plot generation failed")
            raise



    def generate_local_explanation( self, explainer, shap_values, X_transformed,customer_ids_sample, X_sample, feature_names, customer_index: int):

        try:
            logger.info(
                "Generating local SHAP explanation for customer index: %s",
                customer_index
            )

            if customer_index < 0 or customer_index >= len(X_sample):
                raise IndexError("Customer index is outside the selected sample")

            # Extract individual customer's SHAP values
            customer_shap_values = shap_values[customer_index]

            # Extract transformed feature values
            customer_values = X_transformed[customer_index]

            if hasattr(customer_values, "toarray"):
                customer_values = customer_values.toarray().ravel()

            # Create SHAP explanation object
            explanation = shap.Explanation(
                values=customer_shap_values,
                base_values=explainer.expected_value,
                data=customer_values,
                feature_names=feature_names
            )

            # Generate waterfall plot
            shap.plots.waterfall(
                explanation,
                max_display=self.config.top_n_features,
                show=False
            )

            plt.tight_layout()

            output_dir = (
                self.config.report_output_dir
                / self.config.local_explanation_dir
            )

            output_dir.mkdir(parents=True, exist_ok=True)

            customer_id = customer_ids_sample.iloc[customer_index]
            plot_path = (
                output_dir
                / f"{self.config.local_explanation_prefix}{ customer_id }.png"
            )

            plt.savefig(
                plot_path,
                dpi=300,
                bbox_inches="tight"
            )

            plt.close()

            logger.info("Local SHAP explanation saved: %s", plot_path)

            return explanation

        except Exception:
            logger.exception("Local SHAP explanation failed")
            raise



    def save_shap_values( self, shap_values, feature_names, customer_ids):
        try:
            logger.info("Saving structured SHAP values report")

            shap_df = pd.DataFrame(
                shap_values,
                columns=feature_names
            )

            shap_df.insert(
                0,
                "customerID",
                customer_ids.values
            )

            output_path = (
                self.config.report_output_dir
                / self.config.shap_values_file
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shap_df.to_csv(output_path, index=False)

            logger.info(
                "SHAP values report saved: %s",
                output_path
            )

            return output_path

        except Exception:
            logger.exception("Failed to save SHAP values report")
            raise



    def generate_customer_level_shap_report( self, shap_values, feature_names, X_transformed, customer_ids):


        try:
            logger.info("Generating customer-level SHAP report")

            report_rows = []

            for customer_index in range(len(shap_values)):

                customer_shap_values = shap_values[customer_index]

                # Select top features by absolute SHAP contribution
                top_indices = np.argsort(
                    np.abs(customer_shap_values)
                )[::-1][:self.config.top_features_per_customer]

                for feature_index in top_indices:

                    shap_value = customer_shap_values[feature_index]

                    report_rows.append({
                        "customerID": customer_ids.iloc[customer_index],
                        "feature": feature_names[feature_index],
                        "shap_value": shap_value,
                        "impact": (
                            "Increases churn"
                            if shap_value > 0
                            else "Decreases churn"
                            if shap_value < 0
                            else "No impact"
                        ),
                        "transformed_feature_value": (
                            X_transformed[customer_index, feature_index]
                        )
                    })

            report_df = pd.DataFrame(report_rows)

            output_path = (
                self.config.report_output_dir
                / "customer_level_shap_report.csv"
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            report_df.to_csv(output_path, index=False)

            logger.info(
                "Customer-level SHAP report saved: %s",
                output_path
            )

            logger.info(
                "Customer-level SHAP report shape: %s",
                report_df.shape
            )

            return report_df

        except Exception:
            logger.exception(
                "Customer-level SHAP report generation failed"
            )
            raise



    def map_transformed_to_original_features( self, feature_names, numerical_features, categorical_features):
        """
        Map transformed feature names back to their original dataset columns.
        """
        feature_mapping = {}

        for transformed_feature in feature_names:

            if transformed_feature.startswith("numerical__"):
                original_feature = transformed_feature.replace(
                    "numerical__", "", 1
                )

            elif transformed_feature.startswith("categorical__"):
                encoded_feature = transformed_feature.replace(
                    "categorical__", "", 1
                )

                # Match the original categorical column.
                # Longest match avoids ambiguity if column names overlap.
                matching_features = [
                    column
                    for column in categorical_features
                    if encoded_feature.startswith(column + "_")
                    or encoded_feature == column
                ]

                if not matching_features:
                    raise ValueError(
                        f"Could not map transformed feature: {transformed_feature}"
                    )

                original_feature = max(
                    matching_features,
                    key=len
                )

            else:
                raise ValueError(
                    f"Unknown transformed feature prefix: {transformed_feature}"
                )

            feature_mapping[transformed_feature] = original_feature

        return feature_mapping




    def aggregate_shap_values( self, shap_values, feature_names, X_sample, customer_ids, numerical_features, categorical_features):
        """
        Aggregate transformed SHAP values into original feature-level
        contributions for every customer.
        """

        logger.info("Aggregating SHAP values by original features")

        feature_mapping = self.map_transformed_to_original_features(
            feature_names,
            numerical_features,
            categorical_features
        )

        original_features = list(dict.fromkeys(feature_mapping.values()))

        report_rows = []

        for customer_index in range(len(shap_values)):

            customer_shap_values = shap_values[customer_index]

            for original_feature in original_features:

                # Find all transformed columns belonging to this feature
                transformed_indices = [
                    index
                    for index, transformed_feature in enumerate(feature_names)
                    if feature_mapping[transformed_feature] == original_feature
                ]

                # Sum their SHAP contributions
                aggregated_shap_value = customer_shap_values[
                    transformed_indices
                ].sum()

                # Get the original, untransformed feature value
                original_feature_value = X_sample.iloc[
                    customer_index
                ][original_feature]

                report_rows.append({
                    "customerID": customer_ids.iloc[customer_index],
                    "feature": original_feature,
                    "feature_value": original_feature_value,
                    "shap_value": aggregated_shap_value,
                    "impact": (
                        "Increases churn"
                        if aggregated_shap_value > 0
                        else "Decreases churn"
                        if aggregated_shap_value < 0
                        else "No impact"
                    )
                })

        report_df = pd.DataFrame(report_rows)

        logger.info(
            "Original feature-level SHAP report shape: %s",
            report_df.shape
        )

        return report_df
