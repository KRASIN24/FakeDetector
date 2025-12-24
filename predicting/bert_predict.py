"""
Inference-only script for Fake News BERT classifier
"""


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification




def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict(texts, model, tokenizer, max_length=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}


    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)


    return preds.cpu().tolist(), probs.cpu().tolist()

def load_bert(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict_bert(texts, model, tokenizer, max_length=256):
    """
    Predict class indices for input texts using a BERT model.
    
    Args:
        texts: List of input text strings
        model: Loaded BERT model
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
        
    Returns:
        List of predicted class indices
    """
    # Tokenize input texts
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).tolist()
    
    return preds
