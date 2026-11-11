
import pandas as pd

from telecom_churn.entity.config_entity import FeatureEngineeringConfig
from telecom_churn.utils.logger import get_logger


logger = get_logger(__name__)


class FeatureEngineering:

    def __init__(self, config: FeatureEngineeringConfig):
        self.config = config

    def initiate_feature_engineering(self):

        try:
            logger.info("Feature engineering started")

            # Step 1: Create output directory
            self.config.output_data_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            # Step 2: Define input and output file mappings
            file_mappings = {
                self.config.train_data_file:
                    self.config.train_output_file,

                self.config.validation_data_file:
                    self.config.validation_output_file,

                self.config.test_data_file:
                    self.config.test_output_file
            }

            output_paths = {}

            # Step 3: Read and save each dataset
            for input_file, output_file in file_mappings.items():

                input_path = self.config.input_data_dir / input_file
                output_path = self.config.output_data_dir / output_file

                logger.info("Loading dataset: %s", input_path)

                if not input_path.exists():
                    raise FileNotFoundError(
                        f"Dataset not found: {input_path}"
                    )

                df = pd.read_csv(input_path)

                logger.info(
                    "Dataset loaded. Shape: %s",
                    df.shape
                )

                # Preserve all columns and values
                df.to_csv(output_path, index=False)

                output_paths[input_file] = output_path

                logger.info(
                    "Dataset saved successfully: %s",
                    output_path
                )

            logger.info("Feature engineering completed successfully")

            return output_paths

        except Exception:
            logger.exception("Feature engineering failed")
            raise
