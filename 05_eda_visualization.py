
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Disease category distribution
category_counts = df["source_condition_query"].value_counts()

print("--- DISEASE CATEGORY ANALYSIS ---")
print(category_counts)

# Create visualization
plt.figure(figsize=(12, 7))

sns.barplot(
    x=category_counts.values,
    y=category_counts.index
)

plt.title("Clinical Trial Distribution by Disease Category")
plt.xlabel("Number of Clinical Trials")
plt.ylabel("Disease Category")

plt.tight_layout()

# Save chart
plt.savefig(
    "disease_category_distribution.png",
    dpi=300
)

plt.show()

print("\nDisease category chart saved successfully!")
