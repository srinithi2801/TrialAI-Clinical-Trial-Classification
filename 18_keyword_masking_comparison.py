import matplotlib.pyplot as plt

# Model names
models = [
    "Original Model",
    "Keyword-Masked Model"
]

# Accuracy scores
accuracy = [94.48, 88.53]

# Create chart
plt.figure(figsize=(9, 6))

bars = plt.bar(
    models,
    accuracy,
    width=0.5
)

# Add accuracy labels
for bar, score in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{score:.2f}%",
        ha="center",
        fontsize=12
    )

plt.title(
    "Original vs Keyword-Masked Model Performance"
)

plt.xlabel("Model Type")
plt.ylabel("Accuracy (%)")

plt.ylim(0, 100)

plt.tight_layout()

# Save chart
plt.savefig(
    "keyword_masking_comparison.png",
    dpi=300
)

plt.show()

print("Keyword masking comparison chart generated!")