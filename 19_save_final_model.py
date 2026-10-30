import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Input and target
X = df["processed_summary"]
y = df["source_condition_query"]

# Create final model
final_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )),

    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ))
])

print("--- FINAL MODEL TRAINING ---")

# Train using complete dataset
final_model.fit(X, y)

print("Final model training completed!")

# Save model
joblib.dump(
    final_model,
    "clinical_trial_disease_model.pkl"
)

print("\nFinal model saved successfully!")
print("File: clinical_trial_disease_model.pkl")
