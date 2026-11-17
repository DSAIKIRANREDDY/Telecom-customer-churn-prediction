from pathlib import Path
import pandas as pd

from frontend.services.api_client import predict_batch


csv_path = Path("test_customers.csv")

# Inspect the CSV
df = pd.read_csv(csv_path)

print("Dataset shape:", df.shape)
print("\nDataset columns:")
print(df.columns.tolist())

# Upload CSV and get predictions
with csv_path.open("rb") as file:
    result = predict_batch(file)

print("\nTotal customers:", result["total_customers"])

print("\nFirst five predictions:")
for prediction in result["predictions"][:5]:
    print(prediction)