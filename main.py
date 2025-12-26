import os
import argparse
import pandas as pd

from predicting.bert_predict import load_bert, predict_bert
from predicting.tfidf_predict import predict_articles
from services.news_api_client import UnifiedNewsClient
from training.bert_train import train_bert_model
from training.tfidf_train import train_tfidf_model
from services.news_cache import load_cache, load_cached_articles


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", choices=["api", "cache"], default="cache")
    parser.add_argument("--source", choices=["newsapi", "gnews", "both"], default="both")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dataset", type=str, default="default")
    parser.add_argument("--model", choices=["tfidf", "bert"], default="tfidf")

    args = parser.parse_args()

    client = UnifiedNewsClient(
        gnews_key="138931fb000b552c8bcc28b6e4ad4478",
        use_cache=True,
    )

    # --------------------------------------------------
    # Load articles
    # --------------------------------------------------

    if args.data == "api":
        articles = []

        if args.source in ("newsapi", "both"):
            articles += client.fetch(
                query="elections economy",
                source="newsapi",
                dataset=args.dataset,
                force_refresh=args.refresh,
            )

        if args.source in ("gnews", "both"):
            articles += client.fetch(
                query="elections economy",
                source="gnews",
                dataset=args.dataset,
                force_refresh=args.refresh,
            )

    else:
        articles = load_cached_articles(dataset=args.dataset)

    print(f"Loaded {len(articles)} articles")

    new_articles = [a["text"] for a in articles if a.get("text")]

    if not new_articles:
        print("No usable article texts found.")
        return

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    if args.model == "tfidf":
        if not os.path.exists("models/tfidf"):
            train_tfidf_model()

        preds = predict_articles(new_articles)

    elif args.model == "bert":
        train_df = pd.read_csv("data/processed/news.csv")
        val_df = train_df.sample(frac=0.1, random_state=42)

        if not os.path.exists("models/bert"):
            train_bert_model(train_df=train_df, val_df=val_df)

        model, tokenizer = load_bert("models/bert")
        preds = predict_bert(new_articles, model, tokenizer)

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    for text, pred in zip(new_articles, preds):
        label = "True" if pred == 1 else "Fake"
        print(f"{label}: {text[:200]}{'...' if len(text) > 200 else ''}")


if __name__ == "__main__":
    main()
