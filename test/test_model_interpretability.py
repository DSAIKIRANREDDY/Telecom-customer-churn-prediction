
from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.model_interpretability import ModelInterpretability


def test_model_interpretability():

    config_manager = ConfigurationManager()

    config = config_manager.get_model_interpretability_config()

    model_interpretability = ModelInterpretability(config)

    feature_importance = (
        model_interpretability.initiate_model_interpretability()
    )

    print("\nTop 20 Transformed Feature Importance:")
    print(feature_importance.head(20))


if __name__ == "__main__":
    test_model_interpretability()
