from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.model_evaluation import ModelEvaluation

config = ConfigurationManager().get_model_comparison_config()

evaluation = ModelEvaluation(config)

results = evaluation.initiate_model_evaluation()

print("\nMODEL COMPARISON RESULTS")

for model, metrics in results.items():
    print(f"\n{model}")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")