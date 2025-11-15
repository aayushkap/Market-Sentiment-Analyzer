import numpy as np
import pickle
import json
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pathlib import Path


class SentimentClassifier:
    def __init__(self, model_dir="model"):

        self.model_dir = Path(model_dir)
        self.model = None
        self.tokenizer = None
        self.config = None
        self.label_map = {0: "negative", 1: "neutral", 2: "positive"}

        self.load_model()

    def load_model(self):
        """load model, tokenizer and config"""
        config_path = self.model_dir / "lstm_config.json"
        tokenizer_path = self.model_dir / "lstm_tokenizer.pkl"
        model_path = self.model_dir / "lstm_sentiment_model.keras"

        with open(config_path, "r") as f:
            self.config = json.load(f)

        with open(tokenizer_path, "rb") as f:
            self.tokenizer = pickle.load(f)

        self.model = tf.keras.models.load_model(model_path)

    @staticmethod # does not need class instance
    def clean_text(text):
        """
        clean and preprocess text
        """
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"\S+@\S+", "", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def predict(self, text):
        """
        sentiment for a single text
        """
        cleaned = self.clean_text(text)
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(
            sequence, maxlen=self.config["max_sequence_length"], padding="post"
        )

        pred_proba = self.model.predict(padded, verbose=0)[0]
        pred_class = np.argmax(pred_proba)
        sentiment = self.label_map[pred_class]
        confidence = pred_proba[pred_class]

        return {
            "sentiment": sentiment,
            "confidence": float(confidence),
            "probabilities": {
                "negative": float(pred_proba[0]),
                "neutral": float(pred_proba[1]),
                "positive": float(pred_proba[2]),
            },
        }

    def predict_batch(self, texts):
        """
        sentiment for multiple texts
        """
        cleaned = [self.clean_text(text) for text in texts]
        sequences = self.tokenizer.texts_to_sequences(cleaned)
        padded = pad_sequences(
            sequences, maxlen=self.config["max_sequence_length"], padding="post"
        )

        # no python loops during inference means large batches can be processed instantly
        pred_probas = self.model.predict(padded, verbose=0)

        results = []
        for pred_proba in pred_probas:
            pred_class = np.argmax(pred_proba)
            sentiment = self.label_map[pred_class]
            confidence = pred_proba[pred_class]

            results.append(
                {
                    "sentiment": sentiment,
                    "confidence": float(confidence),
                    "probabilities": {
                        "negative": float(pred_proba[0]),
                        "neutral": float(pred_proba[1]),
                        "positive": float(pred_proba[2]),
                    },
                }
            )

        return results

    def __repr__(self):
        return f"SentimentClassifier(model_dir='{self.model_dir}')"
