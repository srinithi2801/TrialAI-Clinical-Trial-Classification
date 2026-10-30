import pandas as pd
from sklearn.model_selection import train_test_split

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Separate input text and target labels
X = df["processed_summary"]
y = df["source_condition_query"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("--- LOGISTIC REGRESSION MODEL ---")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nData preparation completed!")

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Create machine learning pipeline
model = Pipeline([
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

print("\nTraining Logistic Regression model...")

# Train the model
model.fit(X_train, y_train)

print("\nModel training completed successfully!")

from sklearn.metrics import accuracy_score, classification_report

# Predict disease categories
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n--- MODEL EVALUATION ---")

print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

# Detailed classification report
print("\nClassification Report:")

print(classification_report(y_test, y_pred))