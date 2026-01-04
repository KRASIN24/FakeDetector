import os

import torch
import pandas as pd

from training.bilstm_train import BiLSTMAttentionClassifier, get_device, train_bilstm_model
from models.model_utils import encode

# -----------------------------
# Model path
# -----------------------------
MODEL_PATH = "models/bilstm/bilstm.pt"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print("Model not found. Training a new model...")
    train_bilstm_model(model_path=MODEL_PATH)

# -----------------------------
# Load trained BiLSTM
# -----------------------------
def load_bilstm(model_path=MODEL_PATH):
    device = get_device()
    ckpt = torch.load(model_path, map_location=device)

    stoi = ckpt["stoi"]
    embed_dim = ckpt["embed_dim"]
    hidden_dim = ckpt["hidden_dim"]
    # Use the correct number of classes from the checkpoint
    num_classes = ckpt["model_state"]["fc.bias"].shape[0]

    model = BiLSTMAttentionClassifier(
        vocab_size=len(stoi),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, stoi, device


# -----------------------------
# Predict function
# -----------------------------
def predict_bilstm(texts, max_len=512):
    model, stoi, device = load_bilstm()  # load model
    preds = []

    with torch.no_grad():  # disable gradient for inference
        for text in texts:
            # Encode text and move to device
            x = torch.tensor([encode(text, stoi, max_len)], device=device)
            logits = model(x)
            preds.append(logits.argmax(dim=1).item())

    return preds
