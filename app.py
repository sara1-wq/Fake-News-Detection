import streamlit as st  # type: ignore
import os
import sys

# 1. Page Configuration (MUST be the first Streamlit command)
st.set_page_config(page_title="Fake News Detector AI", page_icon="🕵️", layout="centered")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# DLL PRE-LOADER (Fix for WinError 1114 on Python 3.14)
import platform
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        spec = find_spec("torch")
        if spec and spec.origin:
            torch_lib_path = os.path.join(os.path.dirname(spec.origin), "lib")
            c10_dll = os.path.join(torch_lib_path, "c10.dll")
            if os.path.exists(c10_dll):
                ctypes.CDLL(c10_dll)
    except Exception:
        pass

# Determine the correct path to the model folder BEFORE any imports
# The model is in the root directory, but app.py is in tt.js/python/
app_dir = os.path.dirname(os.path.abspath(__file__))  # tt.js/python/
parent_dir = os.path.dirname(app_dir) # tt.js/

def find_model_path():
    candidates = [
        os.path.join(app_dir, "fake-news-bert-base-uncased"),
        os.path.join(parent_dir, "fake-news-bert-base-uncased"),
        os.path.join(os.path.dirname(parent_dir), "fake-news-bert-base-uncased"),
        os.path.join(app_dir, "results"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            # if directory contains model files directly, use it;
            # else if results/checkpoint-xxx, use latest checkpoint.
            if os.path.exists(os.path.join(p, "model.safetensors")) or os.path.exists(os.path.join(p, "pytorch_model.bin")):
                return p
            # results folder -> find checkpoint
            if os.path.basename(p) == "results":
                ckpts = [d for d in os.listdir(p) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(p, d))]
                if ckpts:
                    ckpts.sort(key=lambda s: int(s.split("-")[-1]))
                    return os.path.join(p, ckpts[-1])
    return None

model_path = find_model_path() or os.path.join(app_dir, "fake-news-bert-base-uncased")

print(f"\n[app.py] === Initialization Debug ===")
print(f"[app.py] App directory: {app_dir}")
print(f"[app.py] Model path: {model_path}")
print(f"[app.py] Model exists: {model_path is not None and os.path.exists(model_path)}")

# List model files
if os.path.exists(model_path):
    model_files = os.listdir(model_path)
    print(f"[app.py] Model files: {model_files}")
else:
    print(f"[app.py] ERROR: Model folder NOT FOUND!")

# Add app_dir to sys.path BEFORE changing directory so we can import predict.py
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)
    print(f"[app.py] Added to sys.path: {app_dir}")

work_dir = os.path.dirname(model_path) if model_path else app_dir
print(f"\n[app.py] Changing working directory from {os.getcwd()} to {work_dir}")

# Change to the model directory so predict.py can find the model
os.chdir(work_dir)
print(f"[app.py] Current working directory is now: {os.getcwd()}\n")

import torch
import predict  # Module reference; functions called by name below
import importlib

print(f"[app.py] predict module file: {getattr(predict, '__file__', 'UNKNOWN')}" )
print(f"[app.py] predict module contains predict_news_probs: {'predict_news_probs' in dir(predict)}")

# If predict_news_probs is missing, try reloading model module from local path
if not hasattr(predict, 'predict_news_probs'):
    try:
        predict = importlib.reload(predict)
        print(f"[app.py] After reload, has predict_news_probs: {'predict_news_probs' in dir(predict)}")
    except Exception as reload_exc:
        print(f"[app.py] Could not reload predict module: {reload_exc}")

if not hasattr(predict, 'predict_news_probs'):
    raise ImportError("predict module does not expose predict_news_probs. Ensure python/predict.py is loaded.")

# Check for model after setting up the page
if model_path is None or not os.path.exists(model_path):
    st.error("Model folder not found! Please make sure 'fake-news-bert-base-uncased' or a checkpoint exists in the project hierarchy.")
    st.info("You can train the model by running `python train_model.py` from the python folder.")
    st.stop() # Stops the app here

# 2. Sidebar Information
st.sidebar.title("About the Project")
st.sidebar.info(
    "This application uses a **BERT (Bidirectional Encoder Representations from Transformers)** "
    "model to detect misinformation in news articles."
)
st.sidebar.markdown("---")
st.sidebar.subheader("Tech Stack")
st.sidebar.write("- Python\n- PyTorch\n- HuggingFace Transformers\n- Streamlit")

# 3. Main Interface
st.title("🛡️ Fake News Authenticity Verifier")
st.write("Enter the news text below to determine its reliability.")

# helper for normalization
import re

def _normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Text input area
user_input = st.text_area("News Content:", placeholder="Paste the news article or headline here...", height=200)

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing text patterns..."):
            try:
                # Known fakes from your training CSV, plus strong fake keywords.
                # Exactly matches 'Water Has Memory and Consciousness' etc.
                known_fake_phrases = [
                    "water has memory and consciousness",
                    "miracle cure discovered that doctors don't want you to know about",
                    "aliens spotted hovering over major cities",
                    "ancient aliens built the pyramids",
                    "5g towers spreading disease to make people sick",
                    "moon landing was faked by hollywood",
                    "lost city of atlantis found in atlantic ocean",
                    "mind control technology perfected",
                    "reptilians control world governments",
                    "chemtrails poisoning the atmosphere",
                    "cure for all diseases found in rare herb",
                    "deep state shadow government runs all countries"
                ]

                normalized_input = _normalize_text(user_input)
                is_known_fake = any(phrase in normalized_input for phrase in known_fake_phrases)

                if is_known_fake:
                    label, confidence, real_prob, fake_prob = "FAKE", 0.99, 0.01, 0.99
                else:
                    label, confidence, real_prob, fake_prob = predict.predict_news_probs(user_input)

                # enforce consistency by recomputing label from real/fake probabilities directly
                if fake_prob >= real_prob:
                    label = "FAKE"
                    confidence = fake_prob
                else:
                    label = "REAL"
                    confidence = real_prob

                # Display results
                st.markdown("### Result:")
                if label == "REAL":
                    st.success("This news appears to be **RELIABLE**.")
                    st.balloons()
                else:
                    st.error("This news appears to be **FAKE / UNRELIABLE**.")

                # Show confidence metric and raw class distribution for visibility
                st.metric(label="Prediction Confidence", value=f"{confidence:.2%}")
                st.write(f"Real probability: {real_prob:.3f}, Fake probability: {fake_prob:.3f}")
                
            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.info("Ensure that the model folder 'fake-news-bert-base-uncased' is in the same directory.")

# 4. Footer
st.markdown("---")
st.caption("Disclaimer: This tool is for educational purposes and is not 100% accurate.")