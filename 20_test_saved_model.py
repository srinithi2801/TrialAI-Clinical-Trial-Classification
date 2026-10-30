import joblib

# Load saved model
model = joblib.load("clinical_trial_disease_model.pkl")

print("--- SAVED MODEL VERIFICATION ---")

print("\nModel loaded successfully!")

# Test clinical trial summary
sample_text = """
This study evaluates a new treatment for patients
with breast cancer. The research examines treatment
effectiveness, patient safety and recovery.
"""

# Predict disease category
prediction = model.predict([sample_text])

# Get prediction probabilities
probabilities = model.predict_proba([sample_text])

print("\nSample Clinical Trial Summary:")
print(sample_text)

print("\nPredicted Disease Category:", prediction[0])

print(
    "Prediction Confidence:",
    round(probabilities.max() * 100, 2),
    "%"
)

print("\nSaved model verification completed!")
