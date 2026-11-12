from telecom_churn.config.configuration import ConfigurationManager

from telecom_churn.components.model_evaluation import (
    FinalModelEvaluation
)


def test_final_model_evaluation():

    config_manager = ConfigurationManager()

    config = config_manager.get_model_evaluation_config()

    evaluation = FinalModelEvaluation(config)

    results = evaluation.initiate_final_model_evaluation()

    print("\nFINAL MODEL EVALUATION RESULTS")

    print("\nModel:", results["model"])
    print("Threshold:", results["threshold"])

    print("\nMetrics:")

    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])


if __name__ == "__main__":
    test_final_model_evaluation()