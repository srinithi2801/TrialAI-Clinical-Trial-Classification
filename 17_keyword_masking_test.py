import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Disease keywords to mask
disease_keywords = [
    r"\bchronic obstructive pulmonary disease\b",
    r"\brheumatoid arthritis\b",
    r"\bsickle cell anemia\b",
    r"\bsickle cell disease\b",
    r"\bbreast cancer\b",
    r"\btype 2 diabetes\b",
    r"\btype ii diabetes\b",
    r"\bcovid 19\b",
    r"\bsars cov 2\b",
    r"\bt2dm\b",
    r"\bcopd\b",
    r"\bglaucoma\b",
    r"\banxiety disorder\b",
    r"\banxiety\b"
]

# Combine keywords into one pattern
pattern = re.compile(
    "|".join(disease_keywords),
    flags=re.IGNORECASE
)

# Create masked text
df["masked_summary"] = df["processed_summary"].apply(
    lambda text: pattern.sub(" ", str(text))
)

# Clean extra spaces
df["masked_summary"] = df["masked_summary"].str.replace(
    r"\s+", " ", regex=True
).str.strip()

# Separate input and target
X = df["masked_summary"]
y = df["source_condition_query"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Build Logistic Regression pipeline
masked_model = Pipeline([
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

# Train model
print("--- KEYWORD-MASKED MODEL TRAINING ---")

masked_model.fit(X_train, y_train)

print("\nMasked model training completed!")

# Evaluate model
y_pred = masked_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n--- KEYWORD-MASKED MODEL EVALUATION ---")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")

print(classification_report(y_test, y_pred))