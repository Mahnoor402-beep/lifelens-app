import os
os.environ["HOME"] = "/tmp"

import streamlit as st
from transformers import pipeline
from PIL import Image
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderClassifier
from torchvision import transforms
from torchvision.models import resnet18
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
    result
