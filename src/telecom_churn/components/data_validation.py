
import json
import pandas as pd

from telecom_churn.entity.config_entity import DataValidationConfig
from telecom_churn.utils.common import read_yaml
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class DataValidation:

    def __init__(self, config: DataValidationConfig):
        self.config = config

    def initiate_data_validation(self):

        logger.info("Data validation started")

        try:
            input_path = self.config.input_data_path
            schema_path = self.config.schema_path

            if not input_path.exists():
                raise FileNotFoundError(
                    f"Input dataset not found: {input_path}"
                )

            if not schema_path.exists():
                raise FileNotFoundError(
                    f"Schema file not found: {schema_path}"
                )

            # Load data and schema
            df = pd.read_csv(input_path, keep_default_na=False)
            schema = read_yaml(schema_path)

            logger.info("Dataset loaded. Shape: %s", df.shape)

            errors = []

            # 1. Dataset should not be empty
            if df.empty:
                errors.append("Dataset is empty")

            # 2. Validate column names
            expected_columns = list(schema.columns.keys())
            actual_columns = list(df.columns)

            missing_columns = sorted(
                set(expected_columns) - set(actual_columns)
            )

            extra_columns = sorted(
                set(actual_columns) - set(expected_columns)
            )

            if missing_columns:
                errors.append(
                    f"Missing columns: {missing_columns}"
                )

            if extra_columns:
                errors.append(
                    f"Unexpected columns: {extra_columns}"
                )

            # 3. Validate data types
            dtype_mismatches = {}

            for column, expected_dtype in schema.columns.items():

                if column not in df.columns:
                    continue

                actual_dtype = str(df[column].dtype)

                # Pandas 2.x/3.x may represent strings differently
                if expected_dtype == "string":
                    valid_dtype = (
                        pd.api.types.is_string_dtype(df[column].dtype)
                        or actual_dtype == "object"
                    )
                else:
                    valid_dtype = actual_dtype == expected_dtype

                if not valid_dtype:
                    dtype_mismatches[column] = {
                        "expected": expected_dtype,
                        "actual": actual_dtype
                    }

            if dtype_mismatches:
                errors.append(
                    f"Data type mismatches: {dtype_mismatches}"
                )

            # 4. Check missing and blank values
            missing_values = {}

            blank_allowed = set(
                schema.get("blank_allowed_columns", [])
            )

            for column in df.columns:

                series = df[column]

                null_count = int(series.isna().sum())

                if pd.api.types.is_string_dtype(series.dtype):
                    blank_count = int(
                        series.astype("string")
                        .str.strip()
                        .eq("")
                        .fillna(False)
                        .sum()
                    )
                else:
                    blank_count = 0

                missing_values[column] = {
                    "null_count": null_count,
                    "blank_count": blank_count
                }

                if column not in blank_allowed:
                    if null_count > 0 or blank_count > 0:
                        errors.append(
                            f"Unexpected missing/blank values in {column}"
                        )

            # 5. Validate unique identifiers
            duplicate_counts = {}

            for column in schema.get("unique_columns", []):

                if column not in df.columns:
                    continue

                duplicates = int(df[column].duplicated().sum())
                duplicate_counts[column] = duplicates

                if duplicates > 0:
                    errors.append(
                        f"Duplicate values in {column}: {duplicates}"
                    )

            # 6. Validate target values
            target = schema.target_column

            invalid_target_values = []

            if target in df.columns:

                allowed_values = set(schema.target_values)
                invalid_target_values = [
                    str(value)
                    for value in df[target].dropna().unique()
                    if value not in allowed_values
                ]
                if invalid_target_values:
                    errors.append(
                        f"Invalid target values: {invalid_target_values}"
                    )

            # Prepare validation report
            validation_status = len(errors) == 0

            report = {
                "validation_status": validation_status,
                "dataset_shape": list(df.shape),
                "missing_columns": missing_columns,
                "extra_columns": extra_columns,
                "dtype_mismatches": dtype_mismatches,
                "missing_values": missing_values,
                "duplicate_counts": duplicate_counts,
                "invalid_target_values": invalid_target_values,
                "errors": errors
            }

            # Create output directory
            self.config.validated_data_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Save validation report
            report_path = (
                self.config.validated_data_dir
                / self.config.validation_status_file
            )

            with open(report_path, "w", encoding="utf-8") as file:
                json.dump(report, file, indent=4)

            logger.info(
                "Validation report saved: %s", report_path
            )

            # Stop pipeline if validation fails
            if not validation_status:
                logger.error(
                    "Data validation failed: %s", errors
                )
                raise ValueError(
                    "Data validation failed. Check validation report."
                )

            # Save validated dataset
            output_path = (
                self.config.validated_data_dir
                / self.config.validated_data_file
            )

            df.to_csv(output_path, index=False)

            logger.info(
                "Validated dataset saved: %s", output_path
            )

            logger.info("Data validation completed successfully")

            return output_path

        except Exception:
            logger.exception("Data validation encountered an error")
            raise
