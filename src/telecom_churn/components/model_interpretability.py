
import pandas as pd
import joblib
import yaml
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance
from telecom_churn.entity.config_entity import ModelInterpretabilityConfig
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class ModelInterpretability:

    def __init__(self, config: ModelInterpretabilityConfig):
        self.config = config

    def initiate_model_interpretability(self):

        try:
            logger.info("Model interpretability started")

            # Load validation dataset
            logger.info("Loading validation dataset")
            validation_df = pd.read_csv(
                self.config.validation_data_path
            )

            # Load trained model
            logger.info("Loading trained model")
            model = joblib.load(self.config.model_path)

            # Load schema
            logger.info("Loading schema")
            with open(self.config.schema_path, "r", encoding="utf-8") as file:
                schema = yaml.safe_load(file)

            # Extract original feature columns
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
                raise ValueError("Invalid target values found in validation data")

            # Extract preprocessor and classifier
            preprocessor = model.named_steps["preprocessor"]
            classifier = model.named_steps["classifier"]

            # Get transformed feature names
            feature_names = preprocessor.get_feature_names_out()

            # Extract feature importance
            logger.info("Extracting feature importance")

            feature_importance = pd.DataFrame({
                "feature": feature_names,
                "importance": classifier.feature_importances_
            })

            # Sort feature importance
            feature_importance = feature_importance.sort_values(
                by="importance",
                ascending=False
            ).reset_index(drop=True)

            logger.info("Feature importance extracted successfully")

            # Further aggregation will be added in the next step
            numerical_features = schema["numerical_columns"]

            categorical_features = [
                col for col in schema["categorical_columns"]
                if col != target_column
            ]
            transformed_feature_importance = feature_importance.copy()
            original_feature_importance = self.aggregate_feature_importance(
                feature_importance,
                numerical_features,
                categorical_features
            )
            self.save_feature_importance_reports( transformed_feature_importance, original_feature_importance)
            self.plot_feature_importance(original_feature_importance)

            permutation_df = self.calculate_permutation_importance( model, X, y)
            self.save_permutation_importance(permutation_df)
            self.plot_permutation_importance(permutation_df)
            print("\nPermutation Feature Importance:")
            print(permutation_df.head(20))

            return original_feature_importance, permutation_df

        except Exception:
            logger.exception("Model interpretability failed")
            raise



    def aggregate_feature_importance(self, feature_importance: pd.DataFrame, numerical_features: list, categorical_features: list) -> pd.DataFrame:

        logger.info("Aggregating feature importance")

        def get_original_feature(feature):

            feature = feature.replace("categorical__", "")
            feature = feature.replace("numerical__", "")

            for col in numerical_features + categorical_features:

                if feature == col or feature.startswith(col + "_"):
                    return col

            return feature

        feature_importance["original_feature"] = (
            feature_importance["feature"].apply(get_original_feature)
        )

        original_feature_importance = (
            feature_importance
            .groupby("original_feature")["importance"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        logger.info("Feature importance aggregation completed")

        return original_feature_importance



    def save_feature_importance_reports(self, transformed_feature_importance: pd.DataFrame, original_feature_importance: pd.DataFrame):

        logger.info("Saving feature importance reports")

        transformed_path = (
            self.config.report_output_dir
            / self.config.transformed_feature_importance_file
        )

        original_path = (
            self.config.report_output_dir
            / self.config.feature_importance_file
        )

        transformed_feature_importance.to_csv(
            transformed_path,
            index=False
        )

        original_feature_importance.to_csv(
            original_path,
            index=False
        )

        logger.info("Transformed feature importance saved: %s", transformed_path)
        logger.info("Original feature importance saved: %s", original_path)



    def plot_feature_importance( self, feature_importance: pd.DataFrame):

        logger.info("Generating feature importance plot")

        top_features = (
            feature_importance
            .head(self.config.top_n_features)
            .sort_values(
                by="importance",
                ascending=True
            )
        )

        plt.figure(figsize=(12, 8))

        plt.barh(
            top_features["original_feature"],
            top_features["importance"]
        )

        plt.xlabel("Feature Importance")
        plt.ylabel("Original Features")
        plt.title("Feature Importance - Tuned XGBoost")

        plt.tight_layout()

        plot_path = (
            self.config.report_output_dir
            / self.config.feature_importance_plot
        )

        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Feature importance plot saved: %s", plot_path)


    def calculate_permutation_importance(self, model, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:

        logger.info("Calculating permutation feature importance")

        result = permutation_importance(
            estimator=model,
            X=X,
            y=y,
            scoring=self.config.permutation_scoring,
            n_repeats=self.config.permutation_n_repeats,
            random_state=self.config.random_state,
            n_jobs=-1
        )

        permutation_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std
        })

        permutation_df = permutation_df.sort_values(
            by="importance_mean",
            ascending=False
        ).reset_index(drop=True)

        logger.info("Permutation feature importance calculated")

        return permutation_df


    def save_permutation_importance(self, permutation_df: pd.DataFrame):

        logger.info("Saving permutation importance report")

        output_path = (
            self.config.report_output_dir
            / self.config.permutation_importance_file
        )

        permutation_df.to_csv(output_path, index=False)

        logger.info("Permutation importance saved: %s", output_path)



    def plot_permutation_importance(self, permutation_df: pd.DataFrame):

        logger.info("Generating permutation importance plot")

        top_features = (
            permutation_df
            .head(self.config.top_n_features)
            .sort_values("importance_mean", ascending=True)
        )

        plt.figure(figsize=(12, 8))

        plt.barh(
            top_features["feature"],
            top_features["importance_mean"],
            xerr=top_features["importance_std"]
        )

        plt.xlabel("Decrease in Average Precision")
        plt.ylabel("Original Features")
        plt.title("Permutation Feature Importance - Tuned XGBoost")

        plt.tight_layout()

        plot_path = (
            self.config.report_output_dir
            / self.config.permutation_importance_plot
        )

        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info("Permutation importance plot saved: %s", plot_path)
