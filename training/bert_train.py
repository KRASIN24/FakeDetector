"""
Training script for Transformer-based Fake News Classifier


- Fine-tunes BERT / RoBERTa using PyTorch Lightning
- Supports binary and multi-class classification
- Uses AdamW, linear scheduler, gradient clipping
- Early stopping on validation loss
- Logs accuracy, precision, recall, F1-score, confusion matrix
- Saves model and tokenizer for inference
"""


import os
import math
import torch
import pandas as pd
from typing import Optional, List, Union
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AutoModel
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix




class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, aux_features=None, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.aux = aux_features
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts.iloc[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)

        if self.aux is not None:
            item["aux"] = torch.tensor(
                self.aux.iloc[idx].values, dtype=torch.float
            )

        return item





class FakeNewsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        tokenizer,
        train_df,
        val_df,
        test_df=None,
        batch_size=16,
        max_length=256,
        aux_features_train=None,
        aux_features_val=None,
        aux_features_test=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_length = max_length
        self.aux_features_train = aux_features_train
        self.aux_features_val = aux_features_val
        self.aux_features_test = aux_features_test

    def setup(self, stage: Optional[str] = None):
        self.train_ds = TextDataset(
            self.train_df.text,
            self.train_df.label,
            self.tokenizer,
            aux_features=self.aux_features_train,
            max_length=self.max_length
        )
        self.val_ds = TextDataset(
            self.val_df.text,
            self.val_df.label,
            self.tokenizer,
            aux_features=self.aux_features_val,
            max_length=self.max_length
        )
        self.test_ds = None
        if self.test_df is not None:
            self.test_ds = TextDataset(
                self.test_df.text,
                self.test_df.label,
                self.tokenizer,
                aux_features=self.aux_features_test,
                max_length=self.max_length
            )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)


class TransformerClassifier(pl.LightningModule):
    def __init__(
            self,
            model_name: str,
            num_labels: int,
            lr: float,
            warmup_steps: int,
            total_steps: int,
            aux_dim: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.aux_dim = aux_dim
        if aux_dim > 0:
            self.aux_proj = nn.Linear(aux_dim, hidden_size)
            classifier_input = hidden_size * 2
        else:
            self.aux_proj = None
            classifier_input = hidden_size

        self.classifier = nn.Linear(classifier_input, num_labels)

        self.val_preds = []
        self.val_targets = []

    def forward(
            self,
            input_ids,
            attention_mask,
            labels=None,
            aux=None,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        pooled = outputs.last_hidden_state[:, 0]  # CLS token

        # Optional aux fusion - only use if model was trained with aux AND aux is provided
        if self.aux_proj is not None and aux is not None:
            aux_emb = self.aux_proj(aux)
            pooled = torch.cat([pooled, aux_emb], dim=1)
        elif self.aux_proj is not None and aux is None:
            # Model was trained with aux but none provided - pad with zeros
            batch_size = pooled.shape[0]
            device = pooled.device
            aux_zeros = torch.zeros(batch_size, self.encoder.config.hidden_size, device=device)
            pooled = torch.cat([pooled, aux_zeros], dim=1)

        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return logits, loss

    def training_step(self, batch, batch_idx):
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            aux=batch.get("aux"),
            labels=batch["labels"],
        )
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, loss = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            aux=batch.get("aux"),
            labels=batch["labels"],
        )

        preds = torch.argmax(logits, dim=1)
        self.val_preds.append(preds.cpu())
        self.val_targets.append(batch["labels"].cpu())
        self.log("val_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        acc = accuracy_score(targets, preds)
        p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="weighted")
        cm = confusion_matrix(targets, preds)

        self.log_dict({"val_acc": acc, "val_f1": f1})
        self.confusion_matrix = cm

        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.lr)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            self.hparams.warmup_steps,
            self.hparams.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def train_bert_model(
        train_df,
        val_df,
        aux_features_train=None,
        aux_features_val=None,
        model_name="roberta-base",
        save_dir="models/bert",
        epochs=3,
        batch_size=16,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Determine aux_dim
    aux_dim = 0
    if aux_features_train is not None:
        aux_dim = aux_features_train.shape[1]

    total_steps = math.ceil(len(train_df) / batch_size) * epochs

    model = TransformerClassifier(
        model_name=model_name,
        num_labels=train_df.label.nunique(),
        lr=2e-5,
        warmup_steps=0,
        total_steps=total_steps,
        aux_dim=aux_dim,  # Pass aux_dim here
    )

    # Initialize the data module with aux features
    dm = FakeNewsDataModule(
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        batch_size=batch_size,
        max_length=256,
        aux_features_train=aux_features_train,
        aux_features_val=aux_features_val,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
    )

    trainer.fit(model, dm)

    # Save the entire PyTorch Lightning model
    os.makedirs(save_dir, exist_ok=True)

    # Save the checkpoint
    checkpoint_path = os.path.join(save_dir, "model.ckpt")
    trainer.save_checkpoint(checkpoint_path)

    # Also save tokenizer
    tokenizer.save_pretrained(save_dir)

    # Save aux_dim as metadata
    import json
    metadata = {
        "aux_dim": aux_dim,
        "num_labels": train_df.label.nunique(),
        "model_name": model_name,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(metadata, f)

    print(f"Model saved to {save_dir}")
    print(f"Aux features: {'enabled' if aux_dim > 0 else 'disabled'} (dim={aux_dim})")

    return model
