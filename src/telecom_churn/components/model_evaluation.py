
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)

from telecom_churn.entity.config_entity import ModelComparisonConfig
from telecom_churn.constants import SCHEMA_FILE_PATH
from telecom_churn.utils.common import read_yaml
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class ModelEvaluation:

    def __init__(self, config: ModelComparisonConfig):
        self.config = config



    def initiate_model_evaluation(self):
        try:
            logger.info("Model comparison started")

            # Step 1: Load validation dataset
            df = pd.read_csv(self.config.validation_data_path)

            logger.info(
                "Validation dataset loaded. Shape: %s",
                df.shape
            )

            # Step 2: Load schema
            schema = read_yaml(SCHEMA_FILE_PATH)

            target_column = schema.target_column
            identifier_columns = schema.identifier_columns

            # Step 3: Separate X and y
            X = df.drop(
                columns=[target_column, *identifier_columns]
            )

            y = df[target_column].map({
                "No": 0,
                "Yes": 1
            })

            if y.isnull().any():
                raise ValueError("Unexpected target values found")

            model_directories = [
                self.config.model_dir,
                self.config.tuned_model_dir
            ]

            model_paths = []

            for directory in model_directories:
                if not directory.exists():
                    raise FileNotFoundError(
                        f"Model directory not found: {directory}"
                    )

                model_paths.extend(directory.glob("*.pkl"))

            if not model_paths:
                raise FileNotFoundError("No model files found")

            # Step 4: Evaluate each trained model
            results = {}

            for model_path in model_paths:

                algorithm = model_path.stem.replace("_model", "")

                logger.info("Evaluating model: %s", algorithm)

                model = joblib.load(model_path)

                # Predicted probabilities
                probabilities = model.predict_proba(X)[:, 1]

                # Binary predictions
                predictions = (
                    probabilities >= self.config.threshold
                ).astype(int)

                # Calculate metrics
                metrics = {
                    "accuracy": accuracy_score(y, predictions),
                    "precision": precision_score(
                        y, predictions, zero_division=0
                    ),
                    "recall": recall_score(
                        y, predictions, zero_division=0
                    ),
                    "f1_score": f1_score(
                        y, predictions, zero_division=0
                    ),
                    "roc_auc": roc_auc_score(y, probabilities),
                    "average_precision": average_precision_score(
                        y, probabilities
                    )
                }

                results[algorithm] = metrics

                logger.info(
                    "%s evaluation results: %s",
                    algorithm,
                    metrics
                )

            if not results:
                raise FileNotFoundError(
                    "No models were evaluated from the configured model directories"
                )

            # Step 5: Save comparison report
            report_path = (
                self.config.report_output_dir
                / self.config.comparison_report_file
            )

            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(results, file, indent=4)

            logger.info("Model comparison report saved: %s", report_path)

            logger.info("Model comparison completed successfully")

            return results

        except Exception:
            logger.exception("Model comparison failed")
            raise
