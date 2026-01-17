import torch
import pandas as pd
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from functools import lru_cache
from tqdm import tqdm

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
            device=0 if self.device == 'cuda' else -1
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
        scores = {label: result['scores'][result['labels'].index(label)] if label in result['labels'] else 0.0 for label
                  in self.labels}
        return scores


class EmotionExtractor:
    def __init__(self, model_name='j-hartmann/emotion-english-distilroberta-base', device=None, batch_size=32):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.pipeline = pipeline(
            'text-classification',
            model=self.model,
            tokenizer=self.tokenizer,
            top_k=None,
            truncation=True,
            device=0 if self.device == 'cuda' else -1,
            batch_size=batch_size  # Enable batch processing
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

    def predict_batch(self, texts):
        """Batch prediction for multiple texts"""
        results = self.pipeline(texts)
        batch_scores = []
        for result in results:
            scores = {item['label'].lower(): item['score'] for item in result}
            # Ensure all labels are present
            for label in self.labels:
                if label not in scores:
                    scores[label] = 0.0
            batch_scores.append(scores)
        return batch_scores


class AuxiliaryFeatureExtractor:
    def __init__(self, stance=True, emotion=True, device=None, batch_size=32):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.use_stance = stance
        self.use_emotion = emotion
        self.batch_size = batch_size
        self.stance_model = StanceDetector(device=self.device) if stance else None
        self.emotion_model = EmotionExtractor(device=self.device, batch_size=batch_size) if emotion else None

    def compute_aux_features(self, texts, targets=None, show_progress=True):
        """
        Compute auxiliary features with batch processing and progress bar

        Args:
            texts: List of text strings
            targets: Optional list of target texts for stance detection
            show_progress: Whether to show progress bar
        """
        features_list = []

        # Process in batches
        num_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        iterator = range(0, len(texts), self.batch_size)

        if show_progress:
            iterator = tqdm(iterator, total=num_batches, desc="Computing aux features")

        for start_idx in iterator:
            end_idx = min(start_idx + self.batch_size, len(texts))
            batch_texts = texts[start_idx:end_idx]
            batch_targets = targets[start_idx:end_idx] if targets else None

            # Process batch
            for i, text in enumerate(batch_texts):
                row_features = {}
                target = batch_targets[i] if batch_targets else None

                if self.use_stance:
                    stance_scores = self.stance_model.predict(text, target)
                    row_features.update({f'stance_{k}': v for k, v in stance_scores.items()})

                features_list.append(row_features)

            # Emotion features - process entire batch at once
            if self.use_emotion:
                emotion_batch_scores = self.emotion_model.predict_batch(batch_texts)
                for i, emotion_scores in enumerate(emotion_batch_scores):
                    features_list[start_idx + i].update({f'emotion_{k}': v for k, v in emotion_scores.items()})

        df = pd.DataFrame(features_list)
        return df