from pathlib import Path

import joblib
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = [
            path.name
            for path in (MODEL_PATH, VECTORIZER_PATH)
            if not path.exists()
        ]
        raise FileNotFoundError(
            f"Missing {', '.join(missing)}. Run `python train_model.py` first."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def get_spam_probability(model, input_vector):
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(input_vector)[0]
    spam_class_index = list(model.classes_).index(1)
    return float(probabilities[spam_class_index])


st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

st.title("Spam Email Detector 📧")
st.write("Type a message, click Predict, and see whether it looks like spam.")

try:
    model, vectorizer = load_artifacts()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

user_input = st.text_area(
    "Enter your message:",
    height=180,
    placeholder="Paste an email or SMS message here...",
)

if st.button("Predict", type="primary"):
    message = user_input.strip()

    if not message:
        st.warning("Please enter a message first.")
        st.stop()

    input_vector = vectorizer.transform([message])
    prediction = model.predict(input_vector)[0]
    spam_probability = get_spam_probability(model, input_vector)

    if prediction == 1:
        st.error("🚨 Spam")
    else:
        st.success("✅ Not spam")

    if spam_probability is not None:
        st.metric("Spam probability", f"{spam_probability:.1%}")
        st.progress(spam_probability)
    else:
        st.info("This model does not provide probability scores.")
