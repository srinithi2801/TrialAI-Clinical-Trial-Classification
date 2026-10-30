
import pandas as pd

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

print("--- PROCESSED DATASET VERIFICATION ---")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing processed summaries:")
print(df["processed_summary"].isnull().sum())

print("\nEmpty processed summaries:")
print(
    df["processed_summary"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

print("\nDisease category count:")
print(df["source_condition_query"].nunique())

print("\nVerification completed!")
