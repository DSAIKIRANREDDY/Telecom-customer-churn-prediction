
from telecom_churn.config.configuration import ConfigurationManager


def test_interpretability_configuration():

    config_manager = ConfigurationManager()

    interpretability_config = (
        config_manager.get_model_interpretability_config()
    )

    explainability_config = (
        config_manager.get_model_explainability_config()
    )

    print("\nModel Interpretability Configuration:")
    print(interpretability_config)

    print("\nModel Explainability Configuration:")
    print(explainability_config)


if __name__ == "__main__":
    test_interpretability_configuration()
