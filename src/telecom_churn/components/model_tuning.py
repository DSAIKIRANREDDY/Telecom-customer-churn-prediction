import optuna
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_val_score

from telecom_churn.entity.config_entity import ModelTuningConfig
from telecom_churn.constants import SCHEMA_FILE_PATH
from telecom_churn.utils.common import read_yaml
from telecom_churn.utils.logger import get_logger

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


logger = get_logger(__name__)


class ModelTuning:

    def __init__(self, config: ModelTuningConfig):
        self.config = config


    def load_data(self):
        logger.info("Loading training and validation datasets")

        train_df = pd.read_csv(self.config.train_data_path)
        validation_df = pd.read_csv(self.config.validation_data_path)

        logger.info("Training dataset shape: %s", train_df.shape)
        logger.info("Validation dataset shape: %s", validation_df.shape)

        return train_df, validation_df


    def load_schema(self):
        logger.info("Loading dataset schema")

        schema = read_yaml(SCHEMA_FILE_PATH)

        target_column = schema.target_column
        identifier_columns = schema.identifier_columns
        numerical_features = schema.numerical_columns

        categorical_features = [
            column
            for column in schema.categorical_columns
            if column != target_column
        ]

        logger.info("Target column: %s", target_column)
        logger.info("Numerical features: %s", numerical_features)
        logger.info("Categorical features: %s", categorical_features)

        return ( target_column, identifier_columns, numerical_features, categorical_features)


    def get_numerical_pipeline(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])


    def get_categorical_pipeline(self):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])


    def get_preprocessor(self, numerical_features, categorical_features):

        numerical_pipeline = self.get_numerical_pipeline()
        categorical_pipeline = self.get_categorical_pipeline()

        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_pipeline, numerical_features),
                ("categorical", categorical_pipeline, categorical_features)
            ],
            remainder="drop"
        )

        return preprocessor


    def get_model(self, algorithm: str, trial: optuna.Trial):

        if algorithm == "logistic_regression":

            C = trial.suggest_float(
                "C",
                0.001,
                100,
                log=True
            )

            class_weight = trial.suggest_categorical(
                "class_weight",
                ["balanced", None]
            )

            return LogisticRegression(
                C=C,
                class_weight=class_weight,
                max_iter=1000,
                random_state=self.config.random_state
            )

        elif algorithm == "xgboost":

            n_estimators = trial.suggest_int(
                "n_estimators", 100, 500
            )

            max_depth = trial.suggest_int(
                "max_depth", 2, 8
            )

            learning_rate = trial.suggest_float(
                "learning_rate", 0.01, 0.3, log=True
            )

            subsample = trial.suggest_float(
                "subsample", 0.6, 1.0
            )

            colsample_bytree = trial.suggest_float(
                "colsample_bytree", 0.6, 1.0
            )

            min_child_weight = trial.suggest_int(
                "min_child_weight", 1, 10
            )

            scale_pos_weight = trial.suggest_float(
                "scale_pos_weight", 1.0, 4.0
            )

            reg_alpha = trial.suggest_float(
                "reg_alpha", 1e-8, 10.0, log=True
            )

            reg_lambda = trial.suggest_float(
                "reg_lambda", 1e-3, 100.0, log=True
            )

            return XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                scale_pos_weight=scale_pos_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                eval_metric="logloss",
                random_state=self.config.random_state,
                n_jobs=1
            )

        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}"
            )


    def objective( self, trial, algorithm, X_train, y_train, numerical_features, categorical_features ):

        logger.info(
            "Starting trial %s for algorithm: %s",
            trial.number,
            algorithm
        )

        # Create preprocessing pipeline
        preprocessor = self.get_preprocessor(
            numerical_features,
            categorical_features
        )

        # Create model using Optuna-selected parameters
        model = self.get_model(algorithm, trial)

        # Combine preprocessing and model
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Stratified cross-validation
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state
        )

        # Evaluate model
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=self.config.scoring,
            n_jobs=1,
            error_score="raise"
        )

        mean_score = scores.mean()

        logger.info(
            "Trial %s completed. Average Precision: %.4f",
            trial.number,
            mean_score
        )

        return mean_score


    def tune_algorithm( self, algorithm, X_train, y_train, numerical_features, categorical_features ):

        logger.info("Starting hyperparameter tuning: %s", algorithm)

        sampler = optuna.samplers.TPESampler( seed=self.config.random_state)

        study = optuna.create_study( direction="maximize", study_name=f"{algorithm}_optimization", sampler=sampler )

        study.optimize(
            lambda trial: self.objective(
                trial,
                algorithm,
                X_train,
                y_train,
                numerical_features,
                categorical_features
            ),
            n_trials=self.config.n_trials
        )

        logger.info("Tuning completed for: %s", algorithm)

        logger.info("Best Average Precision: %.4f", study.best_value)
        logger.info("Best hyperparameters: %s", study.best_params)

        return study


    def build_best_model(self, algorithm: str, params: dict):

        if algorithm == "logistic_regression":

            return LogisticRegression(
                C=params["C"],
                class_weight=params["class_weight"],
                max_iter=1000,
                random_state=self.config.random_state
            )

        elif algorithm == "xgboost":

            return XGBClassifier(
                **params,
                eval_metric="logloss",
                random_state=self.config.random_state,
                n_jobs=1
            )

        else:
            raise ValueError(
                f"Unsupported algorithm: {algorithm}"
            )


    def initiate_model_tuning(self):

        try:
            logger.info("Model tuning started")

            # Load datasets
            train_df, validation_df = self.load_data()

            # Load schema
            (
                target_column,
                identifier_columns,
                numerical_features,
                categorical_features
            ) = self.load_schema()

            # Separate training features and target
            X_train = train_df.drop(
                columns=[target_column, *identifier_columns]
            )

            y_train = train_df[target_column].map({
                "No": 0,
                "Yes": 1
            })

            if y_train.isna().any():
                raise ValueError("Invalid target values in training dataset")

            # Run tuning for configured algorithms
            tuned_models = {}

            for algorithm in self.config.algorithms:

                study = self.tune_algorithm(
                    algorithm,
                    X_train,
                    y_train,
                    numerical_features,
                    categorical_features
                )

                # Retrieve best hyperparameters
                best_params = study.best_params

                logger.info(
                    "Best parameters for %s: %s",
                    algorithm,
                    best_params
                )

                # Rebuild the best model using saved parameters
                best_model = self.build_best_model(
                    algorithm,
                    best_params
                )

                # Build complete pipeline
                preprocessor = self.get_preprocessor(
                    numerical_features,
                    categorical_features
                )

                best_pipeline = Pipeline([
                    ("preprocessor", preprocessor),
                    ("classifier", best_model)
                ])

                # Train using complete training dataset
                best_pipeline.fit(X_train, y_train)

                # Save tuned pipeline
                model_path = (
                    self.config.model_output_dir
                    / f"{algorithm}_tuned_model.pkl"
                )

                joblib.dump(best_pipeline, model_path)

                tuned_models[algorithm] = model_path

                logger.info(
                    "Tuned model saved: %s",
                    model_path
                )

            logger.info("Model tuning completed successfully")

            return tuned_models

        except Exception:
            logger.exception("Model tuning failed")
            raise


if __name__ == "__main__":

    from telecom_churn.config.configuration import ConfigurationManager

    config = ConfigurationManager().get_model_tuning_config()

    tuning = ModelTuning(config)

    tuned_models = tuning.initiate_model_tuning()

    print("\nTuned models:")
    print(tuned_models)