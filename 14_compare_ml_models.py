import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score

# Load dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

X = df["processed_summary"]
y = df["source_condition_query"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Create Logistic Regression pipeline
logistic_model = Pipeline([
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

# Create Naive Bayes pipeline
naive_bayes_model = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2
    )),
    ("classifier", MultinomialNB())
])

# Train both models
print("--- TRAINING LOGISTIC REGRESSION ---")
logistic_model.fit(X_train, y_train)

print("\n--- TRAINING NAIVE BAYES ---")
naive_bayes_model.fit(X_train, y_train)

print("\nBoth models trained successfully!")

# Make predictions
logistic_predictions = logistic_model.predict(X_test)
naive_bayes_predictions = naive_bayes_model.predict(X_test)

# Calculate performance metrics
logistic_accuracy = accuracy_score(
    y_test, logistic_predictions
)

naive_bayes_accuracy = accuracy_score(
    y_test, naive_bayes_predictions
)

logistic_f1 = f1_score(
    y_test, logistic_predictions,
    average="macro"
)

naive_bayes_f1 = f1_score(
    y_test, naive_bayes_predictions,
    average="macro"
)

# Display comparison
comparison = pd.DataFrame({
    "Model": ["Logistic Regression", "Multinomial Naive Bayes"],
    "Accuracy": [logistic_accuracy, naive_bayes_accuracy],
    "Macro F1-Score": [logistic_f1, naive_bayes_f1]
})

print("\n--- MODEL PERFORMANCE COMPARISON ---")
print(comparison.to_string(index=False))

print("\nComparison completed!")