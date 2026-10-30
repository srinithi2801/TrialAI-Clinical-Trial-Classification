
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Select disease categories
categories = df["source_condition_query"].unique()

# Display top 10 words for each category
for category in categories:

    category_data = df[
        df["source_condition_query"] == category
    ]

    all_text = " ".join(
        category_data["processed_summary"].dropna()
    )

    words = [
        word for word in all_text.split()
        if not word.isdigit()
    ]

    word_counts = Counter(words)

    top_words = word_counts.most_common(10)

    print("\n--------------------------------")
    print("Disease Category:", category)
    print("--------------------------------")

    for word, count in top_words:
        print(word, ":", count)

    # Create visualization
    word_df = pd.DataFrame(
        top_words,
        columns=["Word", "Frequency"]
    )

    plt.figure(figsize=(10, 5))

    sns.barplot(
        data=word_df,
        x="Frequency",
        y="Word"
    )

    plt.title(f"Top 10 Words - {category}")
    plt.xlabel("Word Frequency")
    plt.ylabel("Words")

    plt.tight_layout()

    filename = (
        category.replace(" ", "_").replace("-", "_")
        + "_top_words.png"
    )

    plt.savefig(filename, dpi=300)
    plt.show()

print("\nDisease-specific word analysis completed!")
