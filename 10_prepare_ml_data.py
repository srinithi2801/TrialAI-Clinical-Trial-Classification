
import pandas as pd
from sklearn.model_selection import train_test_split

# Load processed dataset
df = pd.read_csv(
    "data_sets/clinical_trials_processed.csv",
    low_memory=False
)

# Define input and target
X = df["processed_summary"]
y = df["source_condition_query"]

print("--- MACHINE LEARNING DATA PREPARATION ---")

print("\nInput shape:")
print(X.shape)

print("\nTarget shape:")
print(y.shape)

print("\nDisease categories:")
print(y.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- TRAIN TEST SPLIT ---")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining category distribution:")
print(y_train.value_counts())

print("\nTesting category distribution:")
print(y_test.value_counts())

print("\nMachine learning data preparation completed!")
