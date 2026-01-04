import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from models.model_utils import build_vocab, load_glove, TextDataset


# -----------------------------
# Device
# -----------------------------
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Attention
# -----------------------------
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_out):
        # lstm_out: [batch, seq_len, hidden*2]
        weights = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(weights, dim=1)
        context = torch.sum(lstm_out * weights.unsqueeze(-1), dim=1)
        return context


# -----------------------------
# Model
# -----------------------------
class BiLSTMAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, aux_dim=0, embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if embeddings is not None:
            self.embedding.weight.data.copy_(embeddings)
            self.embedding.weight.requires_grad = False

        self.aux_dim = aux_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

        if aux_dim > 0:
            # project aux features to embedding size
            self.aux_proj = nn.Linear(aux_dim, embed_dim)
        else:
            self.aux_proj = None

        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        """
        Standard forward using token indices
        """
        emb = self.embedding(x)  # [batch, seq_len, embed_dim]
        lstm_out, _ = self.lstm(emb)
        context = self.attention(lstm_out)
        logits = self.fc(context)
        return logits

    def forward_from_embedding(self, x_emb, aux=None):
        """
        x_emb: [batch, seq_len, embed_dim] or [batch, embed_dim]
        aux: [batch, aux_dim] (optional)
        """
        # if x_emb is 2D (batch, embed_dim), add sequence dim
        if x_emb.dim() == 2:
            x_emb = x_emb.unsqueeze(1)  # [batch, 1, embed_dim]

        # handle aux
        if aux is not None and self.aux_proj is not None:
            # project aux to embed_dim
            aux_emb = self.aux_proj(aux)  # [batch, embed_dim]
            # expand to seq_len
            seq_len = x_emb.size(1)
            aux_emb = aux_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq_len, embed_dim]
            # add to embeddings
            x_emb = x_emb + aux_emb

        # Ensure x_emb has the correct final dimension
        assert x_emb.size(-1) == self.embed_dim, f"Expected embed_dim={self.embed_dim}, got {x_emb.size(-1)}"

        lstm_out, _ = self.lstm(x_emb)
        context = self.attention(lstm_out)
        logits = self.fc(context)
        return logits


# -----------------------------
# Training
# -----------------------------
def train_bilstm_model(
        train_csv="data/processed/news.csv",
        text_col="text",
        label_col="label",
        glove_path="embeddings/glove.6B.100d.txt",
        model_path="models/bilstm/bilstm.pt",
        embed_dim: int = 100,
        hidden_dim: int = 128,
        batch_size: int = 32,
        epochs: int = 5,
        lr: float = 1e-3,
        max_len: int = 512,
):
    device = get_device()

    # Load data
    df = pd.read_csv(train_csv)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].tolist()

    stoi = build_vocab(texts)
    embeddings = load_glove(glove_path, stoi, embed_dim)

    dataset = TextDataset(texts, labels, stoi, max_len)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=(device.type == "cuda")
    )

    model = BiLSTMAttentionClassifier(
        vocab_size=len(stoi),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=len(set(labels)),
        embeddings=embeddings
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch + 1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

    # Save model
    torch.save(
        {
            "model_state": model.state_dict(),
            "stoi": stoi,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim
        },
        model_path
    )

    return "BiLSTM training complete"