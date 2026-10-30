
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="TrialAI | Clinical Trial Classifier",
    page_icon="🧬",
    layout="wide"
)

# ==========================================
# 2. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load("clinical_trial_disease_model.pkl")


model = load_model()

# ==========================================
# 3. SESSION STATE
# ==========================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ==========================================
# 4. CUSTOM CSS
# ==========================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background-color: #F7FAFC;
    color: #17324D;
}

[data-testid="stSidebar"] {
    background-color: #EAF5F7;
}

.block-container {
    padding-top: 2rem;
    max-width: 1200px;
}

.hero {
    background: linear-gradient(120deg, #E0F2FE, #CCFBF1);
    padding: 32px;
    border-radius: 16px;
    margin-bottom: 25px;
}

.hero h1 {
    color: #0F172A;
}

.hero p {
    color: #334155;
}

.metric-card {
    background: white;
    height: 132px;
    box-sizing: border-box;
    padding: 18px 10px;
    border-radius: 12px;
    border: 1px solid #DCE8ED;
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    overflow: hidden;
}

.metric-card h4 {
    color: #0F766E;
    font-size: 17px;
    margin-bottom: 8px;
}

.metric-card p {
    color: #64748B;
    margin: 0;
}

.result-card {
    background: #ECFDF5;
    padding: 24px;
    border-radius: 14px;
    border-left: 5px solid #0D9488;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# 5. SIDEBAR
# ==========================================

with st.sidebar:

    st.title("🧬 TrialAI")

    st.caption("Clinical Research Intelligence")

    st.divider()

    st.subheader("About the Project")

    st.write(
        "An NLP and Machine Learning application "
        "that classifies clinical trial summaries "
        "into eight disease categories."
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text Processing:** TF-IDF")
    st.write("**Training Records:** 60,337")
    st.write("**Evaluated Accuracy:** 94.48%")

    st.divider()

    st.subheader("Technology Stack")

    st.write("🐍 Python")
    st.write("📊 Pandas")
    st.write("🧠 Scikit-learn")
    st.write("📝 NLP")
    st.write("🌐 Streamlit")

    st.divider()

    st.caption("Academic Data Science Project")

# ==========================================
# 6. MAIN HEADER
# ==========================================

st.markdown("""
<div class="hero">

    <h1>🧬 Clinical Trial Disease Classifier</h1>

    <p>
        Intelligent disease-category prediction using
        Natural Language Processing and Machine Learning.
    </p>

</div>
""", unsafe_allow_html=True)

# ==========================================
# 7. DISEASE CATEGORY CARDS
# ==========================================

st.subheader("Disease Categories")

diseases = [
    "Anxiety",
    "Breast Cancer",
    "COPD",
    "COVID-19",
    "Glaucoma",
    "Rheumatoid Arthritis",
    "Sickle Cell Anemia",
    "Type 2 Diabetes"
]

for start in range(0, len(diseases), 4):

    cols = st.columns(4)

    for i, disease in enumerate(diseases[start:start + 4]):

        with cols[i]:

            st.markdown(
                f"""
                <div class="metric-card">
                    <h4>{disease}</h4>
                    <p>Disease category</p>
                </div>
                """,
                unsafe_allow_html=True
            )

st.divider()

# ==========================================
# 8. CLINICAL TRIAL ANALYSIS
# ==========================================

st.subheader("🔬 Clinical Trial Analysis")

clinical_text = st.text_area(
    "Enter Clinical Trial Summary",
    placeholder="Paste your clinical trial summary here...",
    height=200
)

if st.button(
    "Analyze Clinical Trial",
    type="primary",
    use_container_width=True
):

    if clinical_text.strip():

        # Predict disease category
        prediction = model.predict([clinical_text])[0]

        # Calculate prediction probabilities
        probabilities = model.predict_proba([clinical_text])[0]

        confidence = max(probabilities) * 100

        # Save prediction history
        st.session_state.prediction_history.append({
            "Time": datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
            "Disease Category": prediction.title(),
            "Confidence (%)": round(confidence, 2)
        })

        st.success("Prediction completed successfully!")

        # ==================================
        # 9. PREDICTION RESULT CARD
        # ==================================

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="result-card">

                <p style="color:#047857;">
                    Predicted Disease Category
                </p>

                <h2 style="color:#064E3B;">
                    {prediction.title()}
                </h2>

                <h3 style="color:#0F766E;">
                    Confidence: {confidence:.2f}%
                </h3>

            </div>
            """,
            unsafe_allow_html=True
        )

        # ==================================
        # 10. PROBABILITY CHART
        # ==================================

        st.subheader("Disease Category Probabilities")

        probability_df = pd.DataFrame({
            "Disease Category": model.classes_,
            "Probability (%)": probabilities * 100
        })

        probability_df = probability_df.sort_values(
            "Probability (%)",
            ascending=False
        )

        st.bar_chart(
            probability_df.set_index("Disease Category")[
                "Probability (%)"
            ],
            horizontal=True
        )

        st.dataframe(
            probability_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Probability (%)": st.column_config.NumberColumn(
                    "Probability (%)",
                    format="%.2f%%"
                )
            }
        )

    else:

        st.warning("Please enter a clinical trial summary.")

# ==========================================
# 11. PREDICTION HISTORY
# ==========================================

st.divider()

st.subheader("📋 Prediction History")

history = st.session_state.prediction_history

if history:

    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    if st.button("Clear Prediction History"):

        st.session_state.prediction_history = []

        st.rerun()

else:

    st.info(
        "No predictions have been made in this session yet."
    )

# ==========================================
# 12. FOOTER
# ==========================================

st.divider()

st.caption(
    "TrialAI | Academic Machine Learning Demonstration | "
    "Predictions are not medical diagnoses."
)

