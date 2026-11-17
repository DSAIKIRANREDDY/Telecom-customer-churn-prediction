
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from pathlib import Path

from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class CustomerExplainability:

    def __init__(self, model_loader, output_dir: Path = Path("artifacts/reports/explainability/inference"), top_n_features: int = 20):

        self.model_loader = model_loader
        self.output_dir = Path(output_dir)
        self.top_n_features = top_n_features

        if self.model_loader.model is None:
            raise RuntimeError("Model must be loaded first")

        if self.model_loader.explainer is None:
            raise RuntimeError("SHAP explainer must be initialized first")

        self.preprocessor = self.model_loader.preprocessor
        self.explainer = self.model_loader.explainer

        self.expected_features = list(
            self.preprocessor.feature_names_in_
        )

        self.feature_names = self.preprocessor.get_feature_names_out()

        self.numerical_features = list(
            self.preprocessor.transformers_[0][2]
        )

        self.categorical_features = list(
            self.preprocessor.transformers_[1][2]
        )

        self.feature_mapping = self._create_feature_mapping()


    def _create_feature_mapping(self):

        mapping = {}

        for transformed_feature in self.feature_names:

            if transformed_feature.startswith("numerical__"):

                original_feature = transformed_feature.replace(
                    "numerical__", "", 1
                )

            elif transformed_feature.startswith("categorical__"):

                encoded_feature = transformed_feature.replace(
                    "categorical__", "", 1
                )

                matching_features = [
                    column
                    for column in self.categorical_features
                    if encoded_feature.startswith(column + "_")
                    or encoded_feature == column
                ]

                if not matching_features:
                    raise ValueError(
                        f"Unable to map feature: {transformed_feature}"
                    )

                original_feature = max(
                    matching_features,
                    key=len
                )

            else:
                raise ValueError(
                    f"Unknown transformed feature: {transformed_feature}"
                )

            mapping[transformed_feature] = original_feature

        return mapping


    def _prepare_customer(self, customer_data: dict):

        customer_df = pd.DataFrame([customer_data])

        # Remove identifier before model processing
        customer_df = customer_df.drop(
            columns=["customerID"],
            errors="ignore"
        )

        missing_features = [
            col for col in self.expected_features
            if col not in customer_df.columns
        ]

        unexpected_features = [
            col for col in customer_df.columns
            if col not in self.expected_features
        ]

        if missing_features:
            raise ValueError(
                f"Missing features: {missing_features}"
            )

        if unexpected_features:
            raise ValueError(
                f"Unexpected features: {unexpected_features}"
            )

        return customer_df[self.expected_features]


    def _get_positive_class_shap_values(self, transformed_data):

        shap_values = self.explainer.shap_values(
            transformed_data
        )

        # Handle SHAP versions returning one array per class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Handle SHAP output with an explicit class dimension
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return np.asarray(shap_values)


    def _get_base_value(self):

        base_value = np.asarray(
            self.explainer.expected_value
        )

        if base_value.size > 1:
            return float(base_value.reshape(-1)[1])

        return float(base_value.reshape(-1)[0])


    def _aggregate_contributions( self, shap_values, original_customer ):

        rows = []

        for original_feature in dict.fromkeys(
            self.feature_mapping.values()
        ):

            transformed_indices = [
                index
                for index, feature in enumerate(self.feature_names)
                if self.feature_mapping[feature] == original_feature
            ]

            contribution = float(
                shap_values[transformed_indices].sum()
            )

            feature_value = original_customer.iloc[0][
                original_feature
            ]

            rows.append({
                "feature": original_feature,
                "feature_value": feature_value,
                "shap_value": contribution,
                "impact": (
                    "Increases churn"
                    if contribution > 0
                    else "Decreases churn"
                    if contribution < 0
                    else "No impact"
                )
            })

        report = pd.DataFrame(rows)

        report["absolute_shap_value"] = (
            report["shap_value"].abs()
        )

        report = report.sort_values(
            by="absolute_shap_value",
            ascending=False
        ).reset_index(drop=True)

        return report


    def _generate_waterfall_plot( self, shap_values, transformed_data, base_value, customer_id ):

        # Convert sparse matrix to dense array for plotting
        if hasattr(transformed_data, "toarray"):
            customer_values = transformed_data.toarray().ravel()
        else:
            customer_values = np.asarray(
                transformed_data
            ).ravel()

        explanation = shap.Explanation(
            values=shap_values,
            base_values=base_value,
            data=customer_values,
            feature_names=self.feature_names
        )

        plt.figure(figsize=(12, 8))

        shap.plots.waterfall(
            explanation,
            max_display=self.top_n_features,
            show=False
        )

        plt.tight_layout()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        safe_customer_id = re.sub(
            r"[^A-Za-z0-9_-]",
            "_",
            str(customer_id)
        )

        plot_path = (
            self.output_dir /
            f"customer_shap_{safe_customer_id}.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            "Customer SHAP plot saved: %s",
            plot_path
        )

        return plot_path


    def explain_customer(self, customer_data: dict):

        try:

            logger.info("Customer-level explanation started")

            customer_id = customer_data.get(
                "customerID",
                "unknown"
            )

            # Prepare original customer features
            X_customer = self._prepare_customer(
                customer_data
            )

            # Transform using fitted preprocessor
            X_transformed = self.preprocessor.transform(
                X_customer
            )

            # Calculate SHAP values
            shap_values = self._get_positive_class_shap_values(
                X_transformed
            )

            if shap_values.ndim != 2 or shap_values.shape[0] != 1:
                raise ValueError(
                    "Expected SHAP values for exactly one customer"
                )

            customer_shap_values = shap_values[0]

            base_value = self._get_base_value()

            # Aggregate transformed contributions
            contribution_report = self._aggregate_contributions(
                customer_shap_values,
                X_customer
            )

            # Generate local waterfall plot
            plot_path = self._generate_waterfall_plot(
                customer_shap_values,
                X_transformed,
                base_value,
                customer_id
            )

            logger.info(
                "Customer-level explanation completed"
            )

            return {
                "customerID": customer_id,
                "base_value": base_value,
                "shap_values": customer_shap_values,
                "contributions": contribution_report,
                "plot_path": str(plot_path)
            }

        except Exception:

            logger.exception(
                "Customer-level explanation failed"
            )
            raise



    
    def _aggregate_batch_contributions(self, shap_values):

        """
        Aggregate transformed SHAP contributions into original features.

        Input shape: (customers, transformed_features)
        Output shape: (customers, original_features)
        """

        aggregated_values = {}

        for original_feature in dict.fromkeys(
            self.feature_mapping.values()
        ):

            transformed_indices = [
                index
                for index, feature in enumerate(self.feature_names)
                if self.feature_mapping[feature] == original_feature
            ]

            aggregated_values[original_feature] = shap_values[
                :, transformed_indices
            ].sum(axis=1)

        return pd.DataFrame(aggregated_values)




    def _generate_batch_beeswarm_plot(self,shap_values,transformed_data):

        if hasattr(transformed_data, "toarray"):
            transformed_data = transformed_data.toarray()
        else:
            transformed_data = np.asarray(transformed_data)

        # Remove transformer prefixes from feature names
        readable_feature_names = [
            feature.replace("numerical__", "")
                   .replace("categorical__", "")
            for feature in self.feature_names
        ]

        explanation = shap.Explanation(
            values=shap_values,
            data=transformed_data,
            feature_names=readable_feature_names
        )

        plt.figure(figsize=(12, 9))

        shap.plots.beeswarm(explanation,max_display=self.top_n_features,show=False)

        plt.title(
            "Batch Customer SHAP Summary",
            fontsize=14
        )

        plt.tight_layout()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plot_path = (
            self.output_dir / "batch_shap_beeswarm.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            "Batch SHAP beeswarm plot saved: %s",
            plot_path
        )

        return plot_path




    def _generate_batch_feature_importance(self,aggregated_shap_values):

        # Calculate mean absolute SHAP contribution
        importance = (
            aggregated_shap_values.abs().mean()
            .sort_values(ascending=True)
        )

        importance = importance.tail(
            self.top_n_features
        )

        plt.figure(figsize=(11, 8))

        importance.plot(
            kind="barh"
        )

        plt.xlabel("Mean Absolute SHAP Value")
        plt.ylabel("Customer Features")

        plt.title(
            "Batch Customer Feature Importance",
            fontsize=14
        )

        plt.tight_layout()

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        plot_path = (
            self.output_dir / "batch_shap_feature_importance.png"
        )

        plt.savefig(
            plot_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        logger.info(
            "Batch SHAP feature importance plot saved: %s",
            plot_path
        )

        return plot_path




    def explain_batch(self, customers_df: pd.DataFrame):

        try:

            logger.info(
                "Batch-level SHAP explanation started"
            )

            if not isinstance(customers_df, pd.DataFrame):
                raise TypeError(
                    "Input must be a pandas DataFrame"
                )

            if customers_df.empty:
                raise ValueError(
                    "Batch input cannot be empty"
                )

            if customers_df.columns.duplicated().any():
                raise ValueError(
                    "Duplicate input columns found"
                )

            # Remove customer identifier
            X_batch = customers_df.drop(
                columns=["customerID"],
                errors="ignore"
            )

            missing_features = [
                col for col in self.expected_features
                if col not in X_batch.columns
            ]

            unexpected_features = [
                col for col in X_batch.columns
                if col not in self.expected_features
            ]

            if missing_features:
                raise ValueError(
                    f"Missing features: {missing_features}"
                )

            if unexpected_features:
                raise ValueError(
                    f"Unexpected features: {unexpected_features}"
                )

            X_batch = X_batch[self.expected_features]

            # Transform using the existing fitted preprocessor
            X_transformed = self.preprocessor.transform(
                X_batch
            )

            # Calculate SHAP values for all customers
            shap_values = self._get_positive_class_shap_values(
                X_transformed
            )

            if (
                shap_values.ndim != 2
                or shap_values.shape[0] != len(X_batch)
                or shap_values.shape[1] != len(self.feature_names)
            ):
                raise ValueError(
                    "Unexpected batch SHAP output shape"
                )

            # Generate beeswarm plot using transformed features
            beeswarm_path = self._generate_batch_beeswarm_plot(
                shap_values,
                X_transformed
            )

            # Aggregate contributions to original features
            aggregated_shap_values = (
                self._aggregate_batch_contributions(
                    shap_values
                )
            )

            # Generate original-feature importance chart
            importance_path = (
                self._generate_batch_feature_importance(
                    aggregated_shap_values
                )
            )

            # Calculate feature importance report
            importance_report = (
                aggregated_shap_values.abs()
                .mean()
                .sort_values(ascending=False)
                .rename("mean_absolute_shap")
                .reset_index()
            )

            importance_report.rename(
                columns={"index": "feature"},
                inplace=True
            )

            logger.info(
                "Batch-level SHAP explanation completed for %s customers",
                len(X_batch)
            )

            return {
                "total_customers": len(X_batch),
                "beeswarm_plot_path": str(beeswarm_path),
                "feature_importance_plot_path": str(importance_path),
                "feature_importance": importance_report,
                "aggregated_shap_values": aggregated_shap_values
            }

        except Exception:

            logger.exception(
                "Batch-level SHAP explanation failed"
            )

            raise
