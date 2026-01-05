"""
Inference-only script for Fake News BERT classifier
"""
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================================
# Standard HuggingFace Model Functions (no aux features)
# ============================================================================

def load_model(model_dir: str):
    """
    Load a standard HuggingFace model (trained without custom aux features)

    Args:
        model_dir: Directory containing the saved model

    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return model, tokenizer


def predict(texts, model, tokenizer, max_length=256):
    """
    Make predictions with a standard HuggingFace model

    Args:
        texts: List of text strings
        model: HuggingFace model
        tokenizer: Tokenizer
        max_length: Maximum sequence length

    Returns:
        Tuple of (predictions, probabilities)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    enc = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.cpu().tolist(), probs.cpu().tolist()


# ============================================================================
# Custom PyTorch Lightning Model Functions (with optional aux features)
# ============================================================================

def load_bert(model_dir):
    """
    Load a custom PyTorch Lightning model (trained with optional aux features)
    Supports both old (HuggingFace) and new (PyTorch Lightning) model formats

    Args:
        model_dir: Directory containing the saved checkpoint

    Returns:
        Tuple of (model, tokenizer)
    """
    # Load tokenizer (works for both formats)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Check which format the model is saved in
    checkpoint_path = os.path.join(model_dir, "model.ckpt")
    config_path = os.path.join(model_dir, "config.json")

    # NEW FORMAT: PyTorch Lightning checkpoint with aux support
    if os.path.exists(checkpoint_path) and os.path.exists(config_path):
        print("📦 Loading new format model (PyTorch Lightning with aux support)")

        # Import here to avoid circular imports
        try:
            from training.bert_train import TransformerClassifier
        except ImportError:
            # Try alternative import path
            try:
                import sys
                sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                from training.bert_train import TransformerClassifier
            except ImportError:
                raise ImportError(
                    "Cannot import TransformerClassifier. Make sure training.bert_train is in your Python path."
                )

        with open(config_path, "r") as f:
            config = json.load(f)

        model = TransformerClassifier.load_from_checkpoint(
            checkpoint_path,
            model_name=config["model_name"],
            num_labels=config["num_labels"],
            lr=2e-5,  # These don't matter for inference
            warmup_steps=0,
            total_steps=1,
            aux_dim=config["aux_dim"],
        )

        model.eval()

        # Print info about aux features
        if config["aux_dim"] > 0:
            print(f"✓ Model expects auxiliary features (dim={config['aux_dim']})")
            print("  - Providing aux_features: will use them")
            print("  - Not providing aux_features: will pad with zeros (may reduce accuracy)")
        else:
            print("✓ Model was trained without auxiliary features")

    # OLD FORMAT: Standard HuggingFace model (no aux support)
    else:
        print("📦 Loading old format model (HuggingFace, no aux support)")
        print("⚠ Warning: This model does not support auxiliary features.")
        print("   To use aux features, retrain the model with the updated training script.")

        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        model.eval()

        # Add a flag to indicate this is an old model
        model.hparams = type('obj', (object,), {'aux_dim': 0})()

    return model, tokenizer


def predict_bert(texts, model, tokenizer, aux_features=None, max_length=256):
    """
    Make predictions with a BERT model (supports both old and new formats)

    Args:
        texts: List of text strings or single string
        model: TransformerClassifier or AutoModelForSequenceClassification
        tokenizer: Tokenizer
        aux_features: DataFrame or array of auxiliary features (optional)
                     - If model was trained with aux but none provided, will pad with zeros
                     - If model was trained without aux, this parameter is ignored
        max_length: Maximum sequence length

    Returns:
        List of predicted class labels
    """
    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False

    device = next(model.parameters()).device

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Check if this is a new format model (with aux support)
    is_new_format = hasattr(model, 'forward') and 'aux' in model.forward.__code__.co_varnames

    with torch.no_grad():
        if is_new_format:
            # NEW FORMAT: Custom model with aux support
            aux = None
            if hasattr(model.hparams, 'aux_dim') and model.hparams.aux_dim > 0:
                # Model was trained with aux
                if aux_features is not None:
                    aux = torch.tensor(aux_features.values, dtype=torch.float32).to(device)
                else:
                    print(f"⚠ Warning: Model was trained with aux features (dim={model.hparams.aux_dim}), "
                          "but none provided. Using zero padding (may reduce accuracy).")
            else:
                # Model was trained without aux
                if aux_features is not None:
                    print("ℹ Note: Model was trained without aux features. Ignoring provided aux_features.")

            logits, _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                aux=aux,
            )
        else:
            # OLD FORMAT: Standard HuggingFace model
            if aux_features is not None:
                print("ℹ Note: Old format model does not support aux features. Ignoring provided aux_features.")

            outputs = model(**inputs)
            logits = outputs.logits

        preds = torch.argmax(logits, dim=1).tolist()

    # Return single prediction if single input
    if single_input:
        return preds[0]

    return preds


def predict_bert_with_probabilities(texts, model, tokenizer, aux_features=None, max_length=256):
    """
    Make predictions with the trained model and return probabilities
    Supports both old and new model formats

    Args:
        texts: List of text strings or single string
        model: TransformerClassifier or AutoModelForSequenceClassification
        tokenizer: Tokenizer
        aux_features: DataFrame or array of auxiliary features (optional)
        max_length: Maximum sequence length

    Returns:
        Tuple of (predictions, probabilities)
        - predictions: List of predicted class labels (or single int if input was single string)
        - probabilities: List of probability distributions (or single list if input was single string)
    """
    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False

    device = next(model.parameters()).device

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Check if this is a new format model (with aux support)
    is_new_format = hasattr(model, 'forward') and 'aux' in model.forward.__code__.co_varnames

    with torch.no_grad():
        if is_new_format:
            # NEW FORMAT: Custom model with aux support
            aux = None
            if hasattr(model.hparams, 'aux_dim') and model.hparams.aux_dim > 0 and aux_features is not None:
                aux = torch.tensor(aux_features.values, dtype=torch.float32).to(device)

            logits, _ = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                aux=aux,
            )
        else:
            # OLD FORMAT: Standard HuggingFace model
            outputs = model(**inputs)
            logits = outputs.logits

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1).tolist()
        probs_list = probs.cpu().tolist()

    # Return single prediction if single input
    if single_input:
        return preds[0], probs_list[0]

    return preds, probs_list

