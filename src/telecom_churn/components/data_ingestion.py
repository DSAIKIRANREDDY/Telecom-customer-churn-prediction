
import pandas as pd

from telecom_churn.entity.config_entity import DataIngestionConfig
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class DataIngestion:

    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self):

        try:
            logger.info("Data ingestion started")

            # Get source dataset path
            source_path = self.config.source_data_path

            logger.info("Checking source dataset: %s", source_path)

            # Check source file existence
            if not source_path.exists():
                raise FileNotFoundError(
                    f"Source dataset not found: {source_path}"
                )

            # Load source dataset
            logger.info("Loading source dataset")

            df = pd.read_csv(source_path)

            logger.info(
                "Dataset loaded successfully. Shape: %s",
                df.shape
            )

            # Create output directory
            self.config.ingested_data_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Define output path
            output_path = (
                self.config.ingested_data_dir
                / self.config.ingested_data_file
            )

            # Save ingested dataset
            df.to_csv(output_path, index=False)

            logger.info(
                "Ingested dataset saved successfully: %s",
                output_path
            )

            logger.info("Data ingestion completed successfully")

            return output_path

        except Exception:
            logger.exception("Data ingestion failed")
            raise
