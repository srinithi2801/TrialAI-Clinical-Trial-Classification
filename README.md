# 🧬 TrialAI – Clinical Trial Disease Category Classification Using NLP and Machine Learning

## 📌 Project Overview

**TrialAI** is a Natural Language Processing (NLP) and Machine Learning project that classifies clinical trial summaries into eight disease categories.

The project uses text preprocessing, TF-IDF feature extraction, and machine learning classification algorithms to identify disease categories from clinical trial descriptions.

An interactive web application developed using **Streamlit** allows users to enter clinical trial summaries and view predicted disease categories along with model probability estimates.

## 🎯 Project Objectives

- Analyze clinical trial descriptions using NLP.
- Clean and preprocess medical text.
- Perform exploratory data analysis (EDA).
- Extract text features using TF-IDF.
- Build and compare machine learning classification models.
- Evaluate model performance using classification metrics.
- Develop an interactive Streamlit application.

## 🏥 Disease Categories

TrialAI classifies clinical trial summaries into the following eight categories:

1. Anxiety
2. Breast Cancer
3. Chronic Obstructive Pulmonary Disease (COPD)
4. COVID-19
5. Glaucoma
6. Rheumatoid Arthritis
7. Sickle Cell Anemia
8. Type 2 Diabetes

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| NLTK | Text preprocessing |
| Scikit-learn | Machine learning |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| TF-IDF | Text feature extraction |
| Streamlit | Interactive web application |
| Joblib | Model serialization |

## 📊 Dataset Information

| Attribute | Description |
|---|---|
| Total records | 60,337 |
| Original features | 16 |
| Input text column | `brief_summary` |
| Target column | `source_condition_query` |
| Processed text column | `processed_summary` |
| Number of categories | 8 |

The target labels are derived from the dataset's `source_condition_query` column.

**Note:** The original dataset files are not included in this repository. They must be obtained separately from an authorized source to reproduce the complete training workflow.

## ⚙️ Machine Learning Workflow

1. Dataset inspection and understanding
2. Data cleaning and validation
3. Text preprocessing
4. Exploratory data analysis
5. Word frequency analysis
6. TF-IDF vectorization
7. Stratified train-test splitting
8. Logistic Regression model training
9. Multinomial Naive Bayes model training
10. Model evaluation and comparison
11. Confusion matrix analysis
12. Disease keyword analysis
13. Keyword masking experiment
14. Final model training and serialization
15. Streamlit application development

### Text Preprocessing

The following NLP techniques were applied:

- Lowercase conversion
- Special character and punctuation removal
- Whitespace normalization
- Tokenization
- Stopword removal while preserving negation terms such as `no`, `not`, `nor`, and `without`
- Word lemmatization

### Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency) was used to convert clinical trial text into numerical features.

Configuration:

- Maximum features: 5,000
- N-gram range: (1, 2)
- Minimum document frequency: 2

## 🤖 Model Performance

Two machine learning algorithms were evaluated using an 80:20 stratified train-test split.

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| Logistic Regression | 94.48% | 93.46% |
| Multinomial Naive Bayes | 90.52% | 89.55% |

**Selected model: Logistic Regression**

Logistic Regression achieved better accuracy and macro F1-score than Multinomial Naive Bayes on the evaluation dataset.

### Keyword Masking Experiment

An exploratory experiment was conducted by masking selected explicit disease-related keywords in clinical trial summaries.

| Metric | Result |
|---|---:|
| Original model accuracy | 94.48% |
| Keyword-masked accuracy | 88.53% |
| Accuracy difference | 5.95 percentage points |

The experiment indicates that the selected explicit disease keywords contribute to model predictions.

However, the masking list was not exhaustive. This experiment does not establish complete robustness to disease-name removal.

## 🌐 TrialAI Streamlit Application

The application includes:

- Clinical trial summary input
- Predicted disease category
- Model probability estimates
- Disease-category probability visualization
- Prediction history
- Interactive dashboard
- Clear prediction history option

The application uses the same text preprocessing function as the model training workflow to maintain preprocessing consistency.

## 📂 Project Structure

```text
Clinical-Trial-Disease-Classification/
│
├── data_sets/
│   └── (Dataset CSV files – not included in GitHub)
│
├── 01_dataset_inspection.py
├── 02_text_preprocessing.py
├── 03_complete_preprocessing.py
├── 04_verify_processed_data.py
├── 05_eda_visualization.py
├── 06_word_frequency_analysis.py
├── 07_summary_length_analysis.py
├── 08_disease_specific_words.py
├── 09_tfidf_vectorization.py
├── 10_prepare_ml_data.py
├── 11_check_duplicate_trials.py
├── 12_train_logistic_regression.py
├── 13_confusion_matrix.py
├── 14_compare_ml_models.py
├── 15_model_comparison_visualization.py
├── 16_disease_keyword_analysis.py
├── 17_keyword_masking_test.py
├── 18_keyword_masking_comparison.py
├── 19_save_final_model.py
├── 20_test_saved_model.py
│
├── app.py
├── text_preprocessing.py
├── clinical_trial_disease_model.pkl
├── requirements.txt
├── .gitignore
├── README.md
│
└── Visualization/
    ├── confusion_matrix.png
    ├── model_comparison.png
    ├── keyword_masking_comparison.png
    ├── disease_category_distribution.png
    └── Other EDA visualizations
```

## 🚀 Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Navigate to the project directory

```bash
cd Clinical-Trial-Disease-Classification
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your default web browser.

**Note:** The trained model file must be present in the project root directory. The dataset is required separately only when reproducing the preprocessing and model training workflow.

## 🔬 Future Improvements

- Experiment with additional machine learning algorithms.
- Improve classification performance for minority categories.
- Expand disease keyword masking experiments.
- Explore deep learning and transformer-based NLP models.
- Implement model interpretability techniques.
- Deploy the application online.
- Improve prediction explanations.

## ⚠️ Disclaimer

TrialAI is an academic machine learning demonstration. It classifies clinical trial summaries according to dataset-derived categories.

It is not a medical diagnostic system and must not be used for medical diagnosis, treatment decisions, or clinical decision-making.

## 👩‍💻 Author

**P. Srinithi**

Data Science and Machine Learning Portfolio Project

**Project:** Clinical Trial Disease Category Classification Using NLP and Machine Learning

**Application:** TrialAI