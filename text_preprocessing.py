
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources if missing
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Load English stopwords
stop_words = set(stopwords.words("english"))

# Preserve important negation words
stop_words -= {"no", "not", "nor", "without"}

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):

    # Handle empty input
    if text is None:
        return ""

    # Convert text to lowercase
    text = str(text).lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    # Return processed text
    return " ".join(tokens)
