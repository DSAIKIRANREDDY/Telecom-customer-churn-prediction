
from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.model_tuning import ModelTuning


def test_model_tuning():
    config_manager = ConfigurationManager()
    config = config_manager.get_model_tuning_config()

    tuning = ModelTuning(config)
    tuning.initiate_model_tuning()


if __name__ == "__main__":
    test_model_tuning()
