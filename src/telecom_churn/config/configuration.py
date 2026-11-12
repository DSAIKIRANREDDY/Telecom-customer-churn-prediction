
from pathlib import Path

from telecom_churn.constants import PROJECT_ROOT, CONFIG_FILE_PATH
from telecom_churn.utils.common import read_yaml, create_directories

from telecom_churn.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataPreprocessingConfig,
    FeatureEngineeringConfig,
    ModelComparisonConfig,  
    ModelTrainingConfig,
    ModelTuningConfig,
    ModelEvaluationConfig,
    MLflowConfig
)


class ConfigurationManager:

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH
    ):

        self.config = read_yaml(config_filepath)

        # Create common artifact directories
        create_directories([
            PROJECT_ROOT / self.config.artifacts.root_dir,
            PROJECT_ROOT / self.config.artifacts.models_dir,
            PROJECT_ROOT / self.config.artifacts.preprocessors_dir,
            PROJECT_ROOT / self.config.artifacts.reports_dir
        ])

    def _resolve_path(self, path: str) -> Path:
        """
        Convert a relative project path into an absolute path.
        """

        path = Path(path)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Create and return data ingestion configuration.
        """

        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            source_data_path=self._resolve_path(
                config.source_data_path
            ),
            ingested_data_dir=self._resolve_path(
                config.ingested_data_dir
            ),
            ingested_data_file=config.ingested_data_file
        )

        create_directories([
            data_ingestion_config.ingested_data_dir
        ])

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:

        config = self.config.data_validation

        validation_config = DataValidationConfig(
            input_data_path=self._resolve_path(config.input_data_path),
            schema_path=self._resolve_path(config.schema_path),
            validated_data_dir=self._resolve_path(config.validated_data_dir),
            validated_data_file=config.validated_data_file,
            validation_status_file=config.validation_status_file
        )

        create_directories([
            validation_config.validated_data_dir
        ])

        return validation_config


    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:

        config = self.config.data_preprocessing

        preprocessing_config = DataPreprocessingConfig(
            input_data_path=self._resolve_path(
                config.input_data_path
            ),
            preprocessed_data_dir=self._resolve_path(
                config.preprocessed_data_dir
            ),
            train_data_file=config.train_data_file,
            validation_data_file=config.validation_data_file,
            test_data_file=config.test_data_file,
            test_size=config.test_size,
            validation_size=config.validation_size,
            random_state=config.random_state,
            target_column=config.target_column
        )

        create_directories([
            preprocessing_config.preprocessed_data_dir
        ])

        return preprocessing_config



    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:

        config = self.config.feature_engineering

        feature_config = FeatureEngineeringConfig(
            input_data_dir=self._resolve_path(config.input_data_dir),
            output_data_dir=self._resolve_path(config.output_data_dir),

            train_data_file=config.train_data_file,
            validation_data_file=config.validation_data_file,
            test_data_file=config.test_data_file,

            train_output_file=config.train_output_file,
            validation_output_file=config.validation_output_file,
            test_output_file=config.test_output_file
        )

        create_directories([
            feature_config.output_data_dir
        ])

        return feature_config


    
    def get_model_training_config(self) -> ModelTrainingConfig:

        config = self.config.model_training

        training_config = ModelTrainingConfig(
            train_data_path=self._resolve_path(config.train_data_path),
            model_output_dir=self._resolve_path(config.model_output_dir),
            random_state=config.random_state,
            algorithms=config.algorithms
        )

        create_directories([
            training_config.model_output_dir
        ])

        return training_config


 
    def get_model_comparison_config(self) -> ModelComparisonConfig:

        config = self.config.model_comparison

        comparison_config = ModelComparisonConfig(
            validation_data_path=self._resolve_path(
                config.validation_data_path
            ),
            model_dir=self._resolve_path(config.model_dir),
            report_output_dir=self._resolve_path(
                config.report_output_dir
            ),
            comparison_report_file=config.comparison_report_file,
            threshold=config.threshold
        )

        create_directories([
            comparison_config.report_output_dir
        ])

        return comparison_config


    
    def get_model_tuning_config(self) -> ModelTuningConfig:

        config = self.config.model_tuning

        tuning_config = ModelTuningConfig(
            train_data_path=self._resolve_path(config.train_data_path),
            validation_data_path=self._resolve_path(config.validation_data_path),
            model_output_dir=self._resolve_path(config.model_output_dir),
            best_model_name=config.best_model_name,
            algorithms=config.algorithms,
            n_trials=config.n_trials,
            cv_folds=config.cv_folds,
            scoring=config.scoring,
            random_state=config.random_state
        )

        create_directories([
            tuning_config.model_output_dir
        ])

        return tuning_config


    def get_model_evaluation_config(self) -> ModelEvaluationConfig:

        config = self.config.model_evaluation

        evaluation_config = ModelEvaluationConfig(
            test_data_path=self._resolve_path(config.test_data_path),
            model_path=self._resolve_path(config.model_path),
            report_output_dir=self._resolve_path(config.report_output_dir),

            classification_report_file=config.classification_report_file,
            confusion_matrix_file=config.confusion_matrix_file,
            threshold=config.threshold
        )

        create_directories([
            evaluation_config.report_output_dir
        ])

        return evaluation_config


    def get_mlflow_config(self) -> MLflowConfig:

        config = self.config.mlflow

        return MLflowConfig(
            experiment_name=config.experiment_name,
            tracking_uri=config.tracking_uri
        )
