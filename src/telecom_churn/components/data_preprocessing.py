
import pandas as pd
from sklearn.model_selection import train_test_split

from telecom_churn.entity.config_entity import DataPreprocessingConfig
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class DataPreprocessing:

    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    def initiate_data_preprocessing(self):

        try:
            # Step 1: Load validated dataset
            df = pd.read_csv(self.config.input_data_path)

            logger.info("Validated dataset loaded. Shape: %s", df.shape)

            df["TotalCharges"] = pd.to_numeric( df["TotalCharges"], errors="coerce")
            logger.info(
                "TotalCharges dtype after conversion: %s",
                df["TotalCharges"].dtype
            )
            logger.info(
                "TotalCharges missing values after conversion: %s",
                df["TotalCharges"].isna().sum()
            )

            target_column = self.config.target_column

            # Step 2: Split into train + validation and test
            train_val_df, test_df = train_test_split(
                df,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=df[target_column]
            )

            # Step 3: Split train + validation into train and validation
            train_df, validation_df = train_test_split(
                train_val_df,
                test_size=self.config.validation_size,
                random_state=self.config.random_state,
                stratify=train_val_df[target_column]
            )

            logger.info("Training dataset shape: %s", train_df.shape)
            logger.info("Validation dataset shape: %s", validation_df.shape)
            logger.info("Test dataset shape: %s", test_df.shape)
            for name, dataset in [
                ("Training", train_df),
                ("Validation", validation_df),
                ("Test", test_df)
            ]:
                
                logger.info(
                        "%s churn distribution:\n%s",
                        name,
                        dataset[target_column].value_counts(normalize=True)
                )
            # Step 4: Create output directory
            self.config.preprocessed_data_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Step 5: Define output paths
            train_path = (
                self.config.preprocessed_data_dir
                / self.config.train_data_file
            )

            validation_path = (
                self.config.preprocessed_data_dir
                / self.config.validation_data_file
            )

            test_path = (
                self.config.preprocessed_data_dir
                / self.config.test_data_file
            )

            # Step 6: Save datasets
            train_df.to_csv(train_path, index=False)
            validation_df.to_csv(validation_path, index=False)
            test_df.to_csv(test_path, index=False)

            logger.info("Training dataset saved: %s", train_path)
            logger.info("Validation dataset saved: %s", validation_path)
            logger.info("Test dataset saved: %s", test_path)

            logger.info("Data preprocessing completed successfully")

            return train_path, validation_path, test_path
        except Exception:
            logger.exception("Data preprocessing failed")
            raise
