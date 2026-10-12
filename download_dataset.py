import os
import pandas as pd
import kagglehub

path = kagglehub.dataset_download(
    "harrachimustapha/fifa-world-cup-team-dataset"
)
print("Dataset downloaded to:", path)

train = pd.read_csv(os.path.join(path, "train.csv"))
test = pd.read_csv(os.path.join(path, "test.csv"))

print("\nTrain Columns:")
print(train.columns)

print("\nFirst 5 rows:")
print(train.head())
print("\nColumns:")
print(train.columns.tolist())