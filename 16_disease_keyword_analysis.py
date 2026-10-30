import pandas as pd
import re

# Load original dataset
df = pd.read_csv(
    "data_sets/clinical_trials_raw_patient2trial_conditions.csv",
    low_memory=False
)

# Disease-specific keywords
disease_keywords = {
    "anxiety": [
        r"\banxiety\b",
        r"\banxiety disorder\b"
    ],
    "breast cancer": [
        r"\bbreast cancer\b",
        r"\bbreast carcinoma\b"
    ],
    "chronic obstructive pulmonary disease": [
        r"\bchronic obstructive pulmonary disease\b",
        r"\bCOPD\b"
    ],
    "covid-19": [
        r"\bCOVID\b",
        r"\bSARS-CoV-2\b"
    ],
    "glaucoma": [
        r"\bglaucoma\b"
    ],
    "rheumatoid arthritis": [
        r"\brheumatoid arthritis\b"
    ],
    "sickle cell anemia": [
        r"\bsickle cell\b",
        r"\bsickle cell disease\b"
    ],
    "type 2 diabetes": [
        r"\btype 2 diabetes\b",
        r"\btype II diabetes\b",
        r"\bT2DM\b"
    ]
}

print("--- DISEASE KEYWORD ANALYSIS ---")

results = []

for disease, keywords in disease_keywords.items():

    # Select records belonging to this disease
    disease_df = df[
        df["source_condition_query"] == disease
    ]

    # Check whether the summary contains any matching keyword
    pattern = "|".join(keywords)

    matches = disease_df["brief_summary"].apply(
        lambda text: bool(
            re.search(pattern, str(text), flags=re.IGNORECASE)
        )
    )

    matching_count = matches.sum()
    total_count = len(disease_df)

    percentage = (matching_count / total_count) * 100

    results.append({
        "Disease Category": disease,
        "Total Records": total_count,
        "Keyword Matches": matching_count,
        "Keyword Match Percentage": round(percentage, 2)
    })

# Display results
result_df = pd.DataFrame(results)

print("\n", result_df.to_string(index=False))

# Save results
result_df.to_csv(
    "data_sets/disease_keyword_analysis.csv",
    index=False
)

print("\nKeyword analysis completed!")