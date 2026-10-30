
import pandas as pd

df = pd.read_csv(
    "data_sets/clinical_trials_raw_patient2trial_conditions.csv",
    low_memory=False
)

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

print("\nDuplicate Records:")
print(df.duplicated().sum())

print("\nDisease Category Distribution:")

category_counts = df["source_condition_query"].value_counts()

print(category_counts)

print("\nNumber of Disease Categories:")
print(df["source_condition_query"].nunique())

print("\nDisease Category Percentages:")
print(
    (df["source_condition_query"].value_counts(normalize=True) * 100)
    .round(2)
)

# Inspect clinical trial summaries

print("\n--- BRIEF SUMMARY INSPECTION ---")

# Check missing and empty summaries
print("Missing summaries:", df["brief_summary"].isnull().sum())

print(
    "Empty summaries:",
    df["brief_summary"].astype(str).str.strip().eq("").sum()
)

# Calculate word count
df["summary_word_count"] = (
    df["brief_summary"].astype(str).str.split().str.len()
)

print("\nSummary Word Count Statistics:")
print(df["summary_word_count"].describe())

# Display sample medical summaries
print("\nSample Clinical Trial Summaries:")

for i, summary in enumerate(df["brief_summary"].head(3), start=1):
    print(f"\nSummary {i}:")
    print(summary[:500])
