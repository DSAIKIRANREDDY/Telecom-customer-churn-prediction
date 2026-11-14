from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.model_explainability import ModelExplainability


def test_model_explainability():

    config_manager = ConfigurationManager()

    config = config_manager.get_model_explainability_config()

    model_explainability = ModelExplainability(config)

    result = model_explainability.initiate_model_explainability()

    print("\nOriginal sample shape:", result["X_sample"].shape)
    print("Transformed shape:", result["X_transformed"].shape)
    print("Feature names:", len(result["feature_names"]))
    print("SHAP values shape:", result["shap_values"].shape)
    print("Customer Ids sample shape: ",result["customer_ids_sample"].shape)
    print("\nOriginal feature-level SHAP report shape:", result["original_feature_shap_report"].shape)
    print(result["original_feature_shap_report"].head(20))

if __name__ == "__main__":
    test_model_explainability()
