from predicting.bert_predict import load_bert, predict_bert
from predicting.tfidf_predict import predict_articles
import argparse

from training.bert_train import train_bert_model
from training.tfidf_train import train_tfidf_model
import pandas as pd


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="tfidf",
        choices=["tfidf", "bert"],  # extend later: bert, svm, xgboost, etc.
        help="Model type to train"
    )
    args = parser.parse_args()

    new_articles = [
        "Breaking news: something happened in politics today...",
        "Celebrity scandal goes viral online!"
    ]

    if args.model == "tfidf":
        train_tfidf_model()
        # Predicting

        preds = predict_articles(new_articles)

    elif args.model == "bert":
        train_df = pd.read_csv("data/processed/news.csv")
        # val_df = pd.read_csv("data/val.csv")
        val_df = train_df.sample(frac=0.1, random_state=42)

        train_bert_model(
            train_df=train_df,
            val_df=val_df,
        )
        model, tokenizer = load_bert("models/bert")
        preds = predict_bert(new_articles, model, tokenizer)

    else:
        raise ValueError(f"Unsupported model type: {args.model}")

    for article, pred in zip(new_articles, preds):
        label = "True" if pred == 1 else "Fake"
        print(f"{label}: {article}")



if __name__ == "__main__":
    main()


