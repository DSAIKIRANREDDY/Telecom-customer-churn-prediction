
import pandas as pd

from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class ChurnPredictor:

    def __init__(self, model_loader, threshold: float = 0.65, low_risk_threshold: float = 0.30):

        self.model_loader = model_loader
        self.threshold = threshold
        self.low_risk_threshold = low_risk_threshold

        if not 0 < self.threshold < 1:
            raise ValueError("Threshold must be between 0 and 1")

        if not 0 <= self.low_risk_threshold < self.threshold:
            raise ValueError(
                "Low risk threshold must be between 0 and prediction threshold"
            )

        if self.model_loader.model is None:
            raise RuntimeError("Model must be loaded before prediction")

        self.model = self.model_loader.model

        # Original input columns expected by the fitted preprocessor
        self.expected_features = list(
            self.model_loader.preprocessor.feature_names_in_
        )


    def _prepare_input(self, data: pd.DataFrame):

        if not isinstance(data, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")

        if data.empty:
            raise ValueError("Input data cannot be empty")

        if data.columns.duplicated().any():
            raise ValueError("Duplicate input columns found")

        data = data.copy()

        # Preserve customer ID separately
        customer_ids = (
            data.pop("customerID")
            if "customerID" in data.columns
            else None
        )

        missing_features = [
            col for col in self.expected_features
            if col not in data.columns
        ]

        unexpected_features = [
            col for col in data.columns
            if col not in self.expected_features
        ]

        if missing_features:
            raise ValueError(
                f"Missing required features: {missing_features}"
            )

        if unexpected_features:
            raise ValueError(
                f"Unexpected input features: {unexpected_features}"
            )

        # Maintain the exact feature order used during training
        data = data[self.expected_features]

        return data, customer_ids


    def _get_risk_level(self, probability: float):

        if probability < self.low_risk_threshold:
            return "Low"

        elif probability < self.threshold:
            return "Medium"

        return "High"


    def _predict(self, data: pd.DataFrame):

        X, customer_ids = self._prepare_input(data)

        logger.info("Generating churn predictions")

        probabilities = self.model.predict_proba(X)[:, 1]

        predictions = (
            probabilities >= self.threshold
        ).astype(int)

        results = pd.DataFrame({
            "churn_probability": probabilities,
            "predicted_churn": [
                "Yes" if prediction == 1 else "No"
                for prediction in predictions
            ],
            "risk_level": [
                self._get_risk_level(probability)
                for probability in probabilities
            ]
        })

        if customer_ids is not None:
            results.insert(
                0,
                "customerID",
                customer_ids.values
            )

        logger.info(
            "Predictions generated successfully for %s customers",
            len(results)
        )

        return results


    def predict_single(self, customer_data: dict):

        logger.info("Single customer prediction started")

        input_df = pd.DataFrame([customer_data])

        results = self._predict(input_df)

        return results.iloc[0].to_dict()


    def predict_batch(self, customers_df: pd.DataFrame):

        logger.info("Batch prediction started")

        results = self._predict(customers_df)

        return results
