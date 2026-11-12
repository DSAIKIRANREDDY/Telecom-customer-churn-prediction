
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from telecom_churn.constants import SCHEMA_FILE_PATH
from telecom_churn.entity.config_entity import ThresholdOptimizationConfig
from telecom_churn.utils.common import read_yaml
from telecom_churn.utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdOptimization:

    def __init__(self, config: ThresholdOptimizationConfig):
        self.config = config

    def initiate_threshold_optimization(self):

        try:
            logger.info("Threshold optimization started")

            # 1. Load validation dataset
            df = pd.read_csv(self.config.validation_data_path)

            logger.info(
                "Validation dataset loaded. Shape: %s",
                df.shape
            )

            # 2. Load schema
            schema = read_yaml(SCHEMA_FILE_PATH)

            target_column = schema.target_column
            identifier_columns = schema.identifier_columns

            # 3. Separate features and target
            X = df.drop(
                columns=[target_column, *identifier_columns]
            )

            y = df[target_column].map({
                "No": 0,
                "Yes": 1
            })

            if y.isnull().any():
                raise ValueError("Unexpected target values found")

            # 4. Load tuned model
            model = joblib.load(self.config.model_path)

            logger.info(
                "Loaded model: %s",
                self.config.model_path
            )

            # 5. Generate churn probabilities
            probabilities = model.predict_proba(X)[:, 1]

            results = {}
            thresholds = np.arange(0.1, 0.91, 0.01)
            # 6. Evaluate different thresholds
            for threshold in thresholds:

                predictions = (
                    probabilities >= threshold
                ).astype(int)

                accuracy = accuracy_score(y, predictions)

                precision = precision_score(
                    y, predictions, zero_division=0
                )

                recall = recall_score(
                    y, predictions, zero_division=0
                )

                f1 = f1_score(
                    y, predictions, zero_division=0
                )

                tn, fp, fn, tp = confusion_matrix(
                    y, predictions, labels=[0, 1]
                ).ravel()

                results[str(threshold)] = {
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "confusion_matrix": {
                        "true_negative": int(tn),
                        "false_positive": int(fp),
                        "false_negative": int(fn),
                        "true_positive": int(tp)
                    }
                }

                logger.info(
                    "Threshold: %.2f | Precision: %.4f | "
                    "Recall: %.4f | F1: %.4f",
                    threshold, precision, recall, f1
                )

            # 7. Select threshold with highest F1-score
            best_threshold = max(
                results,
                key=lambda threshold: results[threshold]["f1_score"]
            )

            report = {
                "best_threshold": float(best_threshold),
                "best_metrics": results[best_threshold],
                "threshold_results": results
            }

            # 8. Save report
            self.config.report_output_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            report_path = (
                self.config.report_output_dir /
                self.config.report_file
            )

            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=4)

            logger.info(
                "Threshold optimization report saved: %s",
                report_path
            )

            logger.info("Threshold optimization completed")

            return report

        except Exception:
            logger.exception("Threshold optimization failed")
            raise
