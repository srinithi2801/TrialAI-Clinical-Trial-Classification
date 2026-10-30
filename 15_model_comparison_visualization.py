import matplotlib.pyplot as plt
import numpy as np

# Model names
models = ["Logistic Regression", "Multinomial Naive Bayes"]

# Performance scores
accuracy = [94.48, 90.52]
macro_f1 = [93.46, 89.55]

# Set bar positions
x = np.arange(len(models))
width = 0.35

# Create chart
plt.figure(figsize=(10, 6))

plt.bar(
    x - width/2,
    accuracy,
    width,
    label="Accuracy"
)

plt.bar(
    x + width/2,
    macro_f1,
    width,
    label="Macro F1-Score"
)

# Add labels
plt.title("Machine Learning Model Performance Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance (%)")

plt.xticks(x, models)
plt.ylim(0, 100)

plt.legend()

# Display values on bars
for i, value in enumerate(accuracy):
    plt.text(i - width/2, value + 0.5, f"{value:.2f}%",
             ha="center")

for i, value in enumerate(macro_f1):
    plt.text(i + width/2, value + 0.5, f"{value:.2f}%",
             ha="center")

plt.tight_layout()

# Save chart
plt.savefig("model_comparison.png", dpi=300)

plt.show()

print("Model comparison chart generated successfully!")