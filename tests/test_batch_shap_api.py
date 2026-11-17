
import pandas as pd

from frontend.services.api_client import (
    explain_batch,
    get_batch_explanation_plot
)

# Load five customers
df = pd.read_csv("test_customers.csv").head(5)

# Convert DataFrame into an in-memory CSV upload
class TestUpload:
    name = "test_customers.csv"

    def getvalue(self):
        return df.to_csv(index=False).encode("utf-8")


uploaded_file = TestUpload()

# Generate batch SHAP explanation
result = explain_batch(uploaded_file)

print("\nBatch SHAP result:")
print(result)

# Retrieve beeswarm plot
beeswarm = get_batch_explanation_plot("beeswarm")

print("\nBeeswarm image bytes:", len(beeswarm))

# Retrieve feature importance plot
importance = get_batch_explanation_plot("importance")

print("Importance image bytes:", len(importance))
