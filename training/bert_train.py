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

from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix




class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length


    def __len__(self):
        return len(self.texts)


    def __getitem__(self, idx):
        enc = self.tokenizer(
        str(self.texts.iloc[idx]),
        truncation=True,
        padding='max_length',
        max_length=self.max_length,
        return_tensors='pt'
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        return item




class FakeNewsDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, train_df, val_df, test_df=None, batch_size=16, max_length=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_length = max_length


    def setup(self, stage: Optional[str] = None):
        self.train_ds = TextDataset(self.train_df.text, self.train_df.label, self.tokenizer, self.max_length)
        self.val_ds = TextDataset(self.val_df.text, self.val_df.label, self.tokenizer, self.max_length)
        self.test_ds = None
        if self.test_df is not None:
            self.test_ds = TextDataset(self.test_df.text, self.test_df.label, self.tokenizer, self.max_length)


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)


    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)


    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)




class TransformerClassifier(pl.LightningModule):
    def __init__(self, model_name: str, num_labels: int, lr: float, warmup_steps: int, total_steps: int):
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.val_preds = []
        self.val_targets = []


    def forward(self, **inputs):
        return self.model(**inputs)


    def training_step(self, batch, batch_idx):
        out = self(**batch)
        preds = torch.argmax(out.logits, dim=1)
        self.log('train_loss', out.loss)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        preds = torch.argmax(out.logits, dim=1)
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(batch["labels"].detach().cpu())
        self.log("val_loss", out.loss)
        return out.loss

    def on_validation_epoch_end(self):
        # Concatenate all outputs
        preds = torch.cat(self.val_preds)
        targets = torch.cat(self.val_targets)

        acc = accuracy_score(targets, preds)
        p, r, f1, _ = precision_recall_fscore_support(targets, preds, average="weighted")
        cm = confusion_matrix(targets, preds)

        # log metrics
        self.log_dict({"val_acc": acc, "val_f1": f1})

        # store confusion matrix if needed
        self.confusion_matrix = cm

        # reset for next epoch
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
    model_name="roberta-base",
    save_dir="models/bert",
    epochs=3,
    batch_size=16,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # build datasets / datamodule here
    # dm = FakeNewsDataModule(...)

    total_steps = math.ceil(len(train_df) / batch_size) * epochs

    model = TransformerClassifier(
        model_name=model_name,
        num_labels=train_df.label.nunique(),
        lr=2e-5,
        warmup_steps=0,
        total_steps=total_steps,
    )

    # Initialize the data module
    dm = FakeNewsDataModule(
        tokenizer=tokenizer,
        train_df=train_df,
        val_df=val_df,
        batch_size=batch_size,
        max_length=256
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="gpu",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=[pl.callbacks.EarlyStopping(monitor="val_loss", patience=2)],
    )

    trainer.fit(model, dm)

    os.makedirs(save_dir, exist_ok=True)
    model.model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    return model
