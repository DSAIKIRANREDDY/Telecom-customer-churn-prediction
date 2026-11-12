
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    source_data_path: Path
    ingested_data_dir: Path
    ingested_data_file: str


@dataclass(frozen=True)
class DataValidationConfig:
    input_data_path: Path
    schema_path: Path
    validated_data_dir: Path
    validated_data_file: str
    validation_status_file: str

@dataclass(frozen=True)
class DataPreprocessingConfig:
    input_data_path: Path
    preprocessed_data_dir: Path

    train_data_file: str
    validation_data_file: str
    test_data_file: str

    test_size: float
    validation_size: float
    random_state: int

    target_column: str



@dataclass(frozen=True)
class FeatureEngineeringConfig:
    input_data_dir: Path
    output_data_dir: Path

    train_data_file: str
    validation_data_file: str
    test_data_file: str

    train_output_file: str
    validation_output_file: str
    test_output_file: str


@dataclass(frozen=True)
class ModelTrainingConfig:
    train_data_path: Path
    model_output_dir: Path
    random_state: int
    algorithms: list[str]


@dataclass(frozen=True)
class ModelComparisonConfig:
    validation_data_path: Path
    model_dir: Path
    tuned_model_dir: Path
    report_output_dir: Path
    comparison_report_file: str
    threshold: float


@dataclass(frozen=True)
class ModelTuningConfig:
    train_data_path: Path
    validation_data_path: Path
    model_output_dir: Path
    best_model_name: str
    algorithms: list[str]
    n_trials: int
    cv_folds: int
    scoring: str
    random_state: int


@dataclass(frozen=True)
class ThresholdOptimizationConfig:
    validation_data_path: Path
    model_path: Path
    report_output_dir: Path
    report_file: str
    thresholds: list[float]


@dataclass(frozen=True)
class ModelEvaluationConfig:
    test_data_path: Path
    model_path: Path
    report_output_dir: Path

    classification_report_file: str
    confusion_matrix_file: str
    threshold: float


@dataclass(frozen=True)
class MLflowConfig:
    experiment_name: str
    tracking_uri: str
