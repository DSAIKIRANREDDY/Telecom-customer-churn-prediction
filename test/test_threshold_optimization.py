from telecom_churn.config.configuration import ConfigurationManager
from telecom_churn.components.threshold_optimization import (
    ThresholdOptimization
)


def test_threshold_optimization():

    config_manager = ConfigurationManager()

    config = config_manager.get_threshold_optimization_config()

    threshold_optimization = ThresholdOptimization(config)

    results = threshold_optimization.initiate_threshold_optimization()

    print("\nTHRESHOLD OPTIMIZATION RESULTS")

    print("Best Threshold:", results["best_threshold"])

    print("\nBest Metrics:")
    print(results["best_metrics"])

    print("\nAll Threshold Results:")
   
    for threshold, metrics in results["threshold_results"].items():
        print(f"\nThreshold: {threshold}")
        print(metrics)


if __name__ == "__main__":
    test_threshold_optimization()