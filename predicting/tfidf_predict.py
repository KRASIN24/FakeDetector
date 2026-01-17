import os
import pickle
import numpy as np
from scipy.sparse import hstack

from models.auxiliary_features import AuxiliaryFeatureExtractor


def load_tfidf_model(model_dir="models/tfidf"):
    """
    Load TF-IDF model, vectorizer, and config
    Supports both old format (without config) and new format (with config)

    Args:
        model_dir: Directory containing the model files

    Returns:
        Tuple of (model, vectorizer, config)
    """
    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    config_path = os.path.join(model_dir, "config.pkl")

    # Check for old file names (backward compatibility)
    if not os.path.exists(model_path):
        old_model_path = os.path.join(model_dir, "fake_news_model.pkl")
        if os.path.exists(old_model_path):
            model_path = old_model_path

    if not os.path.exists(vectorizer_path):
        old_vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")
        if os.path.exists(old_vectorizer_path):
            vectorizer_path = old_vectorizer_path

    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Load vectorizer
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Load config (if exists - new format)
    config = None
    if os.path.exists(config_path):
        with open(config_path, "rb") as f:
            config = pickle.load(f)
        print(f"📦 Loaded TF-IDF model from {model_dir}")
        if config.get("use_aux"):
            print(f"✓ Model uses auxiliary features (dim={config.get('aux_dim', 0)})")
        else:
            print("✓ Model does NOT use auxiliary features")
    else:
        # Old format - assume no aux
        print(f"📦 Loaded old format TF-IDF model from {model_dir}")
        print("⚠ No config found - assuming model WITHOUT auxiliary features")
        config = {"use_aux": False, "aux_dim": 0}

    return model, vectorizer, config


def predict_articles(texts, aux_features=None, model_dir="models/tfidf"):
    """
    Predict labels for articles using TF-IDF model

    Args:
        texts: List of article texts
        aux_features: DataFrame of auxiliary features (optional)
                     - If model was trained with aux, this should be provided
                     - If model was trained without aux, this is ignored
        model_dir: Directory containing the model

    Returns:
        List of predictions (0 = Fake, 1 = True)
    """
    # Load model
    model, vectorizer, config = load_tfidf_model(model_dir)

    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False

    # Vectorize text
    X_tfidf = vectorizer.transform(texts)

    # Handle auxiliary features
    if config.get("use_aux"):
        # Model was trained WITH aux
        if aux_features is not None:
            # Use provided aux features
            X_aux = aux_features.values
            X_combined = hstack([X_tfidf, X_aux])
            print(f"✓ Using provided aux features (shape: {X_aux.shape})")
        else:
            # Compute aux features on the fly
            print("⚠ Model expects aux features but none provided. Computing them now...")
            aux_extractor = AuxiliaryFeatureExtractor(stance=True, emotion=True)
            aux_df = aux_extractor.compute_aux_features(texts)
            X_aux = aux_df.values
            X_combined = hstack([X_tfidf, X_aux])
    else:
        # Model was trained WITHOUT aux
        if aux_features is not None:
            print("ℹ Note: Model does not use aux features. Ignoring provided aux_features.")
        X_combined = X_tfidf

    # Predict
    predictions = model.predict(X_combined)

    # Return single prediction if single input
    if single_input:
        return predictions[0]

    return predictions.tolist()


def predict_with_probabilities(texts, aux_features=None, model_dir="models/tfidf"):
    """
    Predict labels and probabilities for articles

    Args:
        texts: List of article texts
        aux_features: DataFrame of auxiliary features (optional)
        model_dir: Directory containing the model

    Returns:
        Tuple of (predictions, probabilities)
    """
    # Load model
    model, vectorizer, config = load_tfidf_model(model_dir)

    # Handle single string input
    if isinstance(texts, str):
        texts = [texts]
        single_input = True
    else:
        single_input = False

    # Vectorize text
    X_tfidf = vectorizer.transform(texts)

    # Handle auxiliary features
    if config.get("use_aux"):
        if aux_features is not None:
            X_aux = aux_features.values
            X_combined = hstack([X_tfidf, X_aux])
        else:
            aux_extractor = AuxiliaryFeatureExtractor(stance=True, emotion=True)
            aux_df = aux_extractor.compute_aux_features(texts)
            X_aux = aux_df.values
            X_combined = hstack([X_tfidf, X_aux])
    else:
        X_combined = X_tfidf

    # Predict
    predictions = model.predict(X_combined)
    probabilities = model.predict_proba(X_combined)

    # Return single prediction if single input
    if single_input:
        return predictions[0], probabilities[0].tolist()

    return predictions.tolist(), probabilities.tolist()


