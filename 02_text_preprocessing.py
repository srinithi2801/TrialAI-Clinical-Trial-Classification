
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv(
    "data_sets/clinical_trials_raw_patient2trial_conditions.csv",
    low_memory=False
)

# Select the medical text column
text = df["brief_summary"].iloc[0]

print("ORIGINAL MEDICAL TEXT:")
print(text)


# Text cleaning function
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply cleaning to one sample
cleaned_text = clean_text(text)

print("\nCLEANED MEDICAL TEXT:")
print(cleaned_text)

# Apply text cleaning to the entire dataset

df["cleaned_summary"] = df["brief_summary"].apply(clean_text)

print("\n--- FULL DATASET CLEANING ---")

print("Original records:", len(df))

print("Cleaned records:", df["cleaned_summary"].count())

print("\nOriginal Summary:")
print(df["brief_summary"].iloc[0])

print("\nCleaned Summary:")
print(df["cleaned_summary"].iloc[0])

# Save cleaned dataset
df.to_csv(
    "data_sets/clinical_trials_cleaned.csv",
    index=False
)

print("\nCleaned dataset saved successfully!")
# Tokenization example

sample_text = df["cleaned_summary"].iloc[0]

tokens = word_tokenize(sample_text)

print("\n--- TOKENIZATION ---")

print("Original cleaned text:")
print(sample_text)

print("\nTokenized words:")
print(tokens)

print("\nTotal tokens:", len(tokens))

# Stopword removal

stop_words = set(stopwords.words("english"))

# Preserve important medical negation words
important_words = {"no", "not", "nor", "without"}

stop_words = stop_words - important_words

sample_tokens = word_tokenize(
    df["cleaned_summary"].iloc[0]
)

filtered_tokens = [
    word for word in sample_tokens
    if word not in stop_words
]

print("\n--- STOPWORD REMOVAL ---")

print("Before stopword removal:")
print(sample_tokens)

print("\nAfter stopword removal:")
print(filtered_tokens)

print("\nOriginal token count:", len(sample_tokens))
print("Filtered token count:", len(filtered_tokens))

# Lemmatization

lemmatizer = WordNetLemmatizer()

lemmatized_tokens = [
    lemmatizer.lemmatize(word)
    for word in filtered_tokens
]

print("\n--- LEMMATIZATION ---")

print("Before lemmatization:")
print(filtered_tokens)

print("\nAfter lemmatization:")
print(lemmatized_tokens)

print("\nOriginal filtered token count:", len(filtered_tokens))
print("Lemmatized token count:", len(lemmatized_tokens))
