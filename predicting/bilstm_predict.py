import os

import torch
import pandas as pd

from training.bilstm_train import BiLSTMAttentionClassifier, get_device, train_bilstm_model
from models.model_utils import encode
import numpy as np

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
def predict_bilstm(texts, aux_features=None, max_len=512):
    """
    texts: list of strings
    aux_features: pd.DataFrame or None, shape [num_texts, aux_dim]
    """
    model, stoi, device = load_bilstm()
    model.eval()

    preds = []

    # convert aux_features to a single torch tensor once
    if aux_features is not None:
        aux_tensor = torch.tensor(
            aux_features.to_numpy(dtype=np.float32),
            device=device
        )
    else:
        aux_tensor = None

    with torch.no_grad():
        for i, text in enumerate(texts):
            # encode text
            x_enc = encode(text, stoi, max_len)
            x_tensor = torch.tensor([x_enc], device=device)  # shape [1, seq_len]

            # get embedding
            x_emb = model.embedding(x_tensor)  # [1, seq_len, embed_dim]

            # get aux row if exists
            aux_row = aux_tensor[i:i+1] if aux_tensor is not None else None  # [1, aux_dim]

            # forward through LSTM + attention
            logits = model.forward_from_embedding(x_emb, aux=aux_row)
            preds.append(logits.argmax(dim=1).item())

    return preds