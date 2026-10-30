
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Calculate summary word count
df["summary_word_count"] = (
    df["brief_summary"].astype(str).str.split().str.len()
)

# Display statistics
print("--- SUMMARY LENGTH ANALYSIS ---")

print(df["summary_word_count"].describe())

print("\nAverage word count:")
print(df["summary_word_count"].mean())

print("\nMedian word count:")
print(df["summary_word_count"].median())

# Create histogram
plt.figure(figsize=(12, 6))

sns.histplot(
    data=df,
    x="summary_word_count",
    bins=50
)

plt.title("Distribution of Clinical Trial Summary Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Number of Clinical Trials")

plt.tight_layout()

plt.savefig(
    "summary_length_distribution.png",
    dpi=300
)

plt.show()

print("\nSummary length analysis completed!")
