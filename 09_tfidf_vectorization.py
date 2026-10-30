
import pandas as pd

# Load processed clinical trial dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nProcessed summary sample:")
print(df["processed_summary"].iloc[0])

print("\nDisease category sample:")
print(df["source_condition_query"].iloc[0])

from sklearn.feature_extraction.text import TfidfVectorizer

# Select input text
X_text = df["processed_summary"].fillna("")

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

# Convert text into numerical features
X_tfidf = tfidf.fit_transform(X_text)

# Display results
print("\n--- TF-IDF VECTORIZATION ---")

print("TF-IDF Matrix Shape:")
print(X_tfidf.shape)

print("\nNumber of features:")
print(len(tfidf.get_feature_names_out()))

print("\nSample feature names:")
print(tfidf.get_feature_names_out()[:20])

print("\nSample TF-IDF values:")
print(X_tfidf[0].toarray()[0][:20])

# Inspect non-zero TF-IDF values for first document

feature_names = tfidf.get_feature_names_out()

first_document = X_tfidf[0]

# Get non-zero feature positions
nonzero_indices = first_document.nonzero()[1]

# Get corresponding TF-IDF scores
feature_scores = [
    (feature_names[index], first_document[0, index])
    for index in nonzero_indices
]

# Sort by TF-IDF score
feature_scores = sorted(
    feature_scores,
    key=lambda x: x[1],
    reverse=True
)

print("\n--- TOP TF-IDF FEATURES ---")

for word, score in feature_scores[:20]:
    print(f"{word}: {score:.4f}")
