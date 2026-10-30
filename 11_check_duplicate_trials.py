import pandas as pd

df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

print("--- DUPLICATE TRIAL CHECK ---")

print("\nTotal records:", len(df))

print(
    "\nDuplicate NCT IDs:",
    df["nct_id"].duplicated().sum()
)

print(
    "\nTrials appearing under multiple disease categories:",
    (df.groupby("nct_id")["source_condition_query"]
       .nunique() > 1).sum()
)