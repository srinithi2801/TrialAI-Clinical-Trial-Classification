
import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv(
    "data_sets/clinical_trials_raw_patient2trial_conditions.csv",
    low_memory=False
)

# Initialize NLP tools
stop_words = set(stopwords.words("english"))

# Preserve medical negation words
stop_words -= {"no", "not", "nor", "without"}

lemmatizer = WordNetLemmatizer()


# Complete preprocessing function
def preprocess_text(text):

    # 1. Lowercase
    text = str(text).lower()

    # 2. Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 3. Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Tokenization
    tokens = word_tokenize(text)

    # 5. Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # 6. Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# Apply preprocessing
print("Starting NLP preprocessing...")

df["processed_summary"] = (
    df["brief_summary"].apply(preprocess_text)
)

# Display results
print("\nOriginal summary:")
print(df["brief_summary"].iloc[0])

print("\nProcessed summary:")
print(df["processed_summary"].iloc[0])

# Save processed dataset
df.to_csv(
    "data_sets/clinical_trials_processed.csv",
    index=False
)

print("\nTotal records processed:", len(df))
print("NLP preprocessing completed!")
print("Processed dataset saved successfully!")
