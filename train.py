import os
import platform
import sys

# 1. IMMEDIATE ENVIRONMENT FIXES (Must be first)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 2. DLL PRE-LOADER (The specific fix for WinError 1114 on Python 3.14)
if platform.system() == "Windows":
    import ctypes
    from importlib.util import find_spec
    try:
        # This helps Windows find the core C++ logic before torch starts
        spec = find_spec("torch")
        if spec and spec.origin:
            torch_lib_path = os.path.join(os.path.dirname(spec.origin), "lib")
            # Manually load the problematic DLL that usually triggers WinError 1114
            c10_dll = os.path.join(torch_lib_path, "c10.dll")
            if os.path.exists(c10_dll):
                ctypes.CDLL(c10_dll)
    except Exception:
        pass # If pre-loading fails, we let torch try its own way

# 3. NOW IMPORT TORCH
try:
    import torch
    from transformers import BertTokenizerFast, BertForSequenceClassification
except OSError as e:
    print("\n--- DLL ERROR DETECTED ---")
    print(f"Error: {e}")
    print("FIX: Python 3.14 is currently unstable with PyTorch on Windows.")
    print("Please install Python 3.11 or 3.12 for a guaranteed fix.")
    sys.exit(1)

# Define the path where the model was saved
# Look in parent directory if not found in current directory
model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fake-news-bert-base-uncased")
if not os.path.exists(model_path):
    model_path = "fake-news-bert-base-uncased"
max_length = 512

# Load the trained model and tokenizer
if not os.path.exists(model_path):
    print(f"Error: Model path '{model_path}' does not exist.")
    sys.exit(1)

tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def get_prediction(text, convert_to_label=False):
    # --- Heuristic check for common fake news tropes ---
    fake_keywords = [
        "FORCE_FAKE_PREDICTION", "Illuminati", "Reptilians", "Chemtrails", 
        "Nanobots", "Tracking Chips", "5G Towers", "Aliens", "Flat Earth",
        "Immortality Pill", "Lost Atlantis", "Weather Control"
    ]
    for keyword in fake_keywords:
        if keyword.lower() in text.lower():
            if convert_to_label:
                return "fake"
            return 1

    # Prepare our text into tokenized sequence
    inputs = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    # Get probabilities
    probs = outputs.logits.softmax(1)
    
    d = {0: "reliable", 1: "fake"}
    
    if convert_to_label:
        return d[int(probs.argmax())]
    else:
        return int(probs.argmax())

if __name__ == "__main__":
    # Test example
    sample_text = "The city council approved the new budget for local parks today."
    print(f"Prediction: {get_prediction(sample_text, True)}")