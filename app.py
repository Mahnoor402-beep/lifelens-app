import os
os.environ["HOME"] = "/tmp"

import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
from torchvision import transforms
import requests
from io import BytesIO

st.set_page_config(page_title="LifeLens: Emotion AI", layout="centered")

st.title("© LifeLens — Multimodal Emotion Detector")
st.markdown("Detect emotions from **💬 Text**, **🎙️ Voice**, and **📸 Face** — all in one place!")

# TEXT EMOTION
text_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

st.subheader("💬 Text Emotion Detection")
text_input = st.text_input("Enter your text here:")
if text_input:
    result = text_classifier(text_input)[0]
    result = sorted(result, key=lambda x: x['score'], reverse=True)
    st.write(f"Top Emotion: **{result[0]['label']}** ({result[0]['score']:.2f})")

# VOICE EMOTION
st.subheader("🎙️ Voice Emotion Detection (WAV/MP3)")
audio_file = st.file_uploader("Upload a voice file", type=["wav", "mp3"])
if audio_file:
    try:
        classifier = EncoderClassifier.from_hparams(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP", savedir="tmpmodel")
        prediction = classifier.classify_file(audio_file)
        st.success(f"Predicted Emotion: **{prediction[3]}**")
    except Exception as e:
        st.error(f"Voice analysis failed: {str(e)}")

# FACE EMOTION
st.subheader("📸 Face Emotion Detection (Image Upload)")
image_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])
if image_file:
    try:
        image = Image.open(image_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Load model
        model = torch.hub.load('nateraw/ferplus', 'resnet18')
        model.eval()

        # Transform
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        img_tensor = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            prediction = torch.argmax(output, dim=1).item()

        emotions = ['neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'contempt']
        st.success(f"Predicted Emotion: **{emotions[prediction]}**")
    except Exception as e:
        st.error(f"Face analysis failed: {str(e)}")
