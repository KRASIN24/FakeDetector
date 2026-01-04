import joblib
import os

import torch
from torch.utils.data import Dataset
from collections import Counter
import numpy as np

def save_model(model, vectorizer, folder='models'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(model, f'{folder}/fake_news_model.pkl')
    joblib.dump(vectorizer, f'{folder}/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved!")

def load_model(folder='models'):
    model = joblib.load(f'{folder}/fake_news_model.pkl')
    vectorizer = joblib.load(f'{folder}/tfidf_vectorizer.pkl')
    return model, vectorizer


def tokenize(text):
    return text.lower().split()




def build_vocab(texts, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    stoi = {"<pad>": 0, "<unk>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            stoi[word] = len(stoi)
    return stoi




def encode(text, stoi, max_len):
    tokens = tokenize(text)
    ids = [stoi.get(t, 1) for t in tokens][:max_len]
    return ids + [0] * (max_len - len(ids))




class TextDataset(Dataset):
    def __init__(self, texts, labels, stoi, max_len):
        self.texts = texts
        self.labels = labels
        self.stoi = stoi
        self.max_len = max_len


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.stoi, self.max_len))
        y = torch.tensor(self.labels[idx])
        return x, y




def load_glove(path, stoi, embed_dim):
    embeddings = np.random.normal(scale=0.6, size=(len(stoi), embed_dim))
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in stoi:
                embeddings[stoi[word]] = np.asarray(parts[1:], dtype=np.float32)
    return torch.tensor(embeddings, dtype=torch.float)