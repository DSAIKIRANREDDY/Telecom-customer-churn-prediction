
import pandas as pd
import joblib
from telecom_churn.entity.config_entity import ModelTrainingConfig
from telecom_churn.constants import SCHEMA_FILE_PATH
from telecom_churn.utils.common import read_yaml
from telecom_churn.utils.logger import get_logger
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


logger = get_logger(__name__)


class ModelTraining:

    def __init__(self, config: ModelTrainingConfig):
        self.config = config

    def initiate_model_training(self):

        try:
            logger.info("Model training started")

            # Step 1: Load training dataset
            df = pd.read_csv(self.config.train_data_path)

            logger.info(
                "Training dataset loaded. Shape: %s",
                df.shape
            )

            # Step 2: Load dataset schema
            schema = read_yaml(SCHEMA_FILE_PATH)

            target_column = schema.target_column
            identifier_columns = schema.identifier_columns

            numerical_features = schema.numerical_columns

            categorical_features = [
                column
                for column in schema.categorical_columns
                if column != target_column
            ]

            # Step 3: Separate input features and target
            X = df.drop(
                columns=[target_column, *identifier_columns]
            )

            y = df[target_column]

            # Step 4: Encode target labels
            y = y.map({
                "No": 0,
                "Yes": 1
            })

            if y.isnull().any():
                raise ValueError(
                    "Unexpected target values found in training data"
                )

            # Step 5: Verify feature groups
            logger.info(
                "Numerical features: %s",
                numerical_features
            )

            logger.info(
                "Categorical features: %s",
                categorical_features
            )

            logger.info("Input features shape: %s", X.shape)
            logger.info("Target shape: %s", y.shape)

            logger.info(
                "Target distribution:\n%s",
                y.value_counts()
            )

            # Step 6: Verify feature groups
            expected_features = (
                set(numerical_features)
                | set(categorical_features)
            )

            if set(X.columns) != expected_features:
                raise ValueError(
                    "Feature mismatch between dataset and schema"
                )

            # Step 7: Build model pipelines
            # Step 8: Train and save all models

            trained_models = {}

            for algorithm in self.config.algorithms:

                logger.info("Training started: %s", algorithm)

                # Build model pipeline
                model_pipeline = self.build_model_pipeline(
                    algorithm,
                    numerical_features,
                    categorical_features
                )

                # Train model
                model_pipeline.fit(X, y)

                logger.info("Training completed: %s", algorithm)

                # Save trained pipeline
                model_path = (
                    self.config.model_output_dir
                    / f"{algorithm}_model.pkl"
                )

                joblib.dump(model_pipeline, model_path)

                trained_models[algorithm] = model_path

                logger.info("Model saved: %s", model_path)

            logger.info("All model training completed successfully")

            return trained_models

        except Exception:
            logger.exception("Model training failed")
            raise


    def get_numerical_pipeline(self):

        numerical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median")
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        return numerical_pipeline


    def get_categorical_pipeline(self):

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent")
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        return categorical_pipeline


    def get_preprocessor(self, numerical_features, categorical_features ):

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    self.get_numerical_pipeline(),
                    numerical_features
                ),
                (
                    "categorical",
                    self.get_categorical_pipeline(),
                    categorical_features
                )
            ],
            remainder="drop"
        )

        return preprocessor

    
    def get_model(self, algorithm):

        if algorithm == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=self.config.random_state
            )

        elif algorithm == "random_forest":
            return RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=self.config.random_state,
                n_jobs=-1
            )

        elif algorithm == "xgboost":
            return XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=self.config.random_state,
                n_jobs=-1
            )

        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}"
            )



    def build_model_pipeline( self, algorithm, numerical_features, categorical_features):

        # Step 1: Create preprocessor
        preprocessor = self.get_preprocessor(
            numerical_features,
            categorical_features
        )

        # Step 2: Create classifier
        classifier = self.get_model(algorithm)

        # Step 3: Combine preprocessor and classifier
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier)
            ]
        )

        return model_pipeline


