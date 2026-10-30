
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Combine processed summaries
all_text = " ".join(df["processed_summary"].dropna())

# Split text into words
words = [
    word for word in all_text.split()
    if not word.isdigit()
]

# Count word frequency
word_counts = Counter(words)

# Get top 20 frequent words
top_words = word_counts.most_common(20)

print("--- TOP 20 FREQUENT WORDS ---")

for word, count in top_words:
    print(word, ":", count)

# Create DataFrame for visualization
word_df = pd.DataFrame(
    top_words,
    columns=["Word", "Frequency"]
)

# Plot
plt.figure(figsize=(12, 7))

sns.barplot(
    data=word_df,
    x="Frequency",
    y="Word"
)

plt.title("Top 20 Most Frequent Words in Clinical Trial Summaries")
plt.xlabel("Frequency")
plt.ylabel("Medical Words")

plt.tight_layout()

plt.savefig("top_20_medical_words.png", dpi=300)
plt.show()

print("\nWord frequency analysis completed!")
