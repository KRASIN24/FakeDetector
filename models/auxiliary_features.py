import torch
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import lru_cache

# Suppress transformers device messages
logging.getLogger("transformers").setLevel(logging.ERROR)

class StanceDetector:
    def __init__(self, model_name='facebook/bart-large-mnli', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.pipeline = pipeline(
            'zero-shot-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device=='cuda' else -1
        )
        self.labels = ['support', 'refute', 'neutral']

    @lru_cache(maxsize=1024)
    def predict(self, text, target=None):
        sequence = text
        candidate_labels = self.labels
        if target:
            # Optionally combine target into sequence for context
            sequence = f"{text} [TARGET]: {target}"
        result = self.pipeline(sequence, candidate_labels)
        # Return probabilities in the order of self.labels
        scores = {label: result['scores'][result['labels'].index(label)] if label in result['labels'] else 0.0 for label in self.labels}
        return scores

class EmotionExtractor:
    def __init__(self, model_name='j-hartmann/emotion-english-distilroberta-base', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.pipeline = pipeline(
            'text-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,  # replacement for deprecated return_all_scores
            device=0 if self.device == 'cuda' else -1
        )
        # Predefined emotion labels based on model
        self.labels = ['anger', 'fear', 'joy', 'love', 'sadness', 'surprise']

    @lru_cache(maxsize=1024)
    def predict(self, text):
        results = self.pipeline(text)[0]  # single text
        scores = {item['label'].lower(): item['score'] for item in results}
        # Ensure all labels are present
        for label in self.labels:
            if label not in scores:
                scores[label] = 0.0
        return scores

class AuxiliaryFeatureExtractor:
    def __init__(self, stance=True, emotion=True, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_stance = stance
        self.use_emotion = emotion
        self.stance_model = StanceDetector(device=self.device) if stance else None
        self.emotion_model = EmotionExtractor(device=self.device) if emotion else None

    def compute_aux_features(self, texts, targets=None):
        features_list = []
        for i, text in enumerate(texts):
            row_features = {}
            target = targets[i] if targets else None
            if self.use_stance:
                stance_scores = self.stance_model.predict(text, target)
                # Prefix keys to avoid collision
                row_features.update({f'stance_{k}': v for k, v in stance_scores.items()})
            if self.use_emotion:
                emotion_scores = self.emotion_model.predict(text)
                row_features.update({f'emotion_{k}': v for k, v in emotion_scores.items()})
            features_list.append(row_features)
        df = pd.DataFrame(features_list)
        return df