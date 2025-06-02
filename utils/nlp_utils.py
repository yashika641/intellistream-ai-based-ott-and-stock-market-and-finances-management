# utils/nlp_utils.py

import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# === Constants === #
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"  # or your fine-tuned model path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model & tokenizer === #
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(DEVICE)

# === Preprocessing === #
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# === Tokenization & Inference === #
def predict_sentiment(text: str) -> dict:
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    label = "positive" if np.argmax(probs) == 1 else "negative"
    return {"label": label, "confidence": float(np.max(probs))}

# === Batch Inference (Optional) === #
def batch_predict(texts: list[str]) -> list[dict]:
    return [predict_sentiment(text) for text in texts]
