import os
import argparse
import pandas as pd

from models.auxiliary_features import AuxiliaryFeatureExtractor
from predicting.bert_predict import load_bert, predict_bert
from predicting.tfidf_predict import predict_articles
from predicting.bilstm_predict import predict_bilstm
from services.news_api_client import UnifiedNewsClient
from training.bert_train import train_bert_model
from training.bilstm_train import train_bilstm_model
from training.tfidf_train import train_tfidf_model
from services.news_cache import load_cache, load_cached_articles



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", choices=["api", "cache"], default="cache")
    parser.add_argument("--source", choices=["newsapi", "gnews", "both"], default="both")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dataset", type=str, default="default")
    parser.add_argument("--model", choices=["tfidf", "bert", "bilstm"], default="tfidf")
    parser.add_argument("--aux", action="store_true", help="Enable auxiliary stance/emotion features")

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
    # Auxiliary features (optional)
    # --------------------------------------------------

    aux_features = None

    if args.aux:
        aux_extractor = AuxiliaryFeatureExtractor(stance=True, emotion=True)
        aux_features = aux_extractor.compute_aux_features(new_articles)
        print("Aux features sample:")
        print(aux_features.head())

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    if args.model == "tfidf":
        if not os.path.exists("models/tfidf"):
            train_tfidf_model()

        preds = predict_articles(new_articles, aux_features=aux_features)

    elif args.model == "bert":
        train_df = pd.read_csv("data/processed/news.csv")
        val_df = train_df.sample(frac=0.1, random_state=42)

        if not os.path.exists("models/bert"):
            train_bert_model(train_df=train_df, val_df=val_df)

        model, tokenizer = load_bert("models/bert")
        preds = predict_bert(new_articles, model, tokenizer, aux_features=aux_features)

    elif args.model == "bilstm":
        train_df = pd.read_csv("data/processed/news.csv")
        val_df = train_df.sample(frac=0.1, random_state=42)

        if not os.path.exists("models/bilstm"):
            train_bilstm_model(train_df=train_df, val_df=val_df)

        preds = predict_bilstm(new_articles, aux_features=aux_features)

    else:
        raise ValueError(f"Unsupported model: {args.model}")

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    stance_cols = ["stance_support", "stance_refute", "stance_neutral"]
    emotion_cols = ["emotion_anger", "emotion_fear", "emotion_joy", "emotion_love", "emotion_sadness",
                    "emotion_surprise"]

    for i, (text, pred) in enumerate(zip(new_articles, preds)):
        label = "True" if pred == 1 else "Fake"
        print(f"\n=== Article {i + 1} ===")
        print(f"Prediction: {label}")
        print(f"Text: {text}")
        # print(f"Text: {text[:300]}{'...' if len(text) > 300 else ''}")

        if aux_features is not None:
            aux_row = aux_features.iloc[i]

            # Stance
            stance_vals = {c.split('_')[-1].capitalize(): round(aux_row[c] * 100, 1) for c in stance_cols}
            stance_str = ", ".join([f"{k}: {v}%" for k, v in stance_vals.items()])
            print(f"Stance: {stance_str}")

            # Emotion
            emotion_vals = {c.split('_')[-1].capitalize(): round(aux_row[c] * 100, 1) for c in emotion_cols}
            emotion_str = ", ".join([f"{k}: {v}%" for k, v in emotion_vals.items()])
            print(f"Emotion: {emotion_str}")


if __name__ == "__main__":
    main()