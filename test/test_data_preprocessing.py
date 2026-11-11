from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.data_preprocessing import DataPreprocessing

config = ConfigurationManager().get_data_preprocessing_config()

preprocessing = DataPreprocessing(config)

preprocessing.initiate_data_preprocessing()