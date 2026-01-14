import streamlit as st
import tempfile
from featureextraction import extract_audio_features, extract_text_features
from utils import speech_to_text
from model import predict_risk

st.set_page_config(page_title="Cognitive Health Screening AI")

st.title("🧠 Speech-Based Cognitive Health Screening")
st.write("This tool screens for early cognitive decline using speech patterns.")

audio = st.file_uploader("Upload a speech sample (30–60 seconds)", type=["wav"])

if audio:
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(audio.read())
        audio_path = temp.name
 
    st.audio(audio)

    with st.spinner("Analyzing speech..."):
        text = speech_to_text(audio_path)
        audio_features = extract_audio_features(audio_path)
        text_features = extract_text_features(text)
        
        combined_features = {**audio_features, **text_features}
        risk_score = predict_risk(combined_features)

    st.subheader("📝 Transcribed Speech")
    st.write(text)

    st.subheader("📊 Cognitive Risk Assessment")
    st.metric("Risk Score", f"{risk_score:.2f}")

    if risk_score > 0.6:
        st.error("⚠️ High Cognitive Risk Detected")
    else:
        st.success("✅ Low Cognitive Risk")

    st.subheader("🔍 Key Indicators")
    st.json(combined_features)

    st.info("⚠️ This is a screening tool, not a medical diagnosis.")
