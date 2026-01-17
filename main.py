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
        print("\n🔧 Computing auxiliary features (stance + emotion)...")
        aux_extractor = AuxiliaryFeatureExtractor(stance=True, emotion=True, batch_size=32)
        aux_features = aux_extractor.compute_aux_features(new_articles, show_progress=True)
        print("Aux features sample:")
        print(aux_features.head())

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    if args.model == "tfidf":
        # TF-IDF uses separate paths for aux/no-aux
        model_dir = "models/tfidf_aux" if args.aux else "models/tfidf"

        print(f"\n📁 Using model directory: {model_dir}")

        # Check if model FILES exist, not just the directory
        model_path = os.path.join(model_dir, "model.pkl")

        if not os.path.exists(model_path):
            print(f"\n🏋️ Training TF-IDF model {'WITH' if args.aux else 'WITHOUT'} auxiliary features...")
            train_tfidf_model(
                save_dir=model_dir,
                aux_features=args.aux
            )
        else:
            print(f"✓ Model already exists at {model_dir}")

        print(f"\n📥 Loading model from {model_dir}")
        preds = predict_articles(new_articles, aux_features=aux_features, model_dir=model_dir)

    elif args.model == "bert":
        # Use separate directories for BERT with/without aux
        model_dir = "models/bert_aux" if args.aux else "models/bert"

        print(f"\n📁 Using model directory: {model_dir}")

        # Load training data
        train_df = pd.read_csv("data/processed/news.csv")
        val_df = train_df.sample(frac=0.1, random_state=42)
        train_df = train_df.drop(val_df.index)  # Remove validation samples from training

        # Check if model needs to be trained
        if not os.path.exists(model_dir) or args.retrain:
            if args.retrain and os.path.exists(model_dir):
                print(f"🗑️  Removing existing model at {model_dir}")
                import shutil
                shutil.rmtree(model_dir)

            print(f"\n🏋️ Training BERT model {'WITH' if args.aux else 'WITHOUT'} auxiliary features...")

            # Compute aux features for training data if --aux is enabled
            aux_features_train = None
            aux_features_val = None

            if args.aux:
                print("🔧 Computing auxiliary features for training data...")
                aux_extractor = AuxiliaryFeatureExtractor(stance=True, emotion=True, batch_size=32)
                aux_features_train = aux_extractor.compute_aux_features(train_df.text.tolist(), show_progress=True)
                aux_features_val = aux_extractor.compute_aux_features(val_df.text.tolist(), show_progress=True)
                print(f"   Training aux features shape: {aux_features_train.shape}")
                print(f"   Validation aux features shape: {aux_features_val.shape}")

            # Train with aux features if enabled
            train_bert_model(
                train_df=train_df,
                val_df=val_df,
                aux_features_train=aux_features_train,
                aux_features_val=aux_features_val,
                save_dir=model_dir,  # Use the appropriate directory
            )
        else:
            print(f"✓ Model already exists at {model_dir}")

        # Load and predict
        print(f"\n📥 Loading model from {model_dir}")
        model, tokenizer = load_bert(model_dir)
        preds = predict_bert(new_articles, model, tokenizer, aux_features=aux_features)

    elif args.model == "bilstm":
        # BiLSTM uses separate paths for aux/no-aux
        model_dir = "models/bilstm_aux" if args.aux else "models/bilstm"

        train_df = pd.read_csv("data/processed/news.csv")
        val_df = train_df.sample(frac=0.1, random_state=42)
        train_df = train_df.drop(val_df.index)

        if not os.path.exists(model_dir) or args.retrain:
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

    print("\n" + "=" * 80)
    print(f"PREDICTIONS - Model: {args.model.upper()}{'_AUX' if args.aux else ''}")
    print("=" * 80)

    for i, (text, pred) in enumerate(zip(new_articles, preds)):
        label = "True" if pred == 1 else "Fake"
        print(f"\n=== Article {i + 1} ===")
        print(f"Prediction: {label}")
        print(f"Text: {text[:200]}{'...' if len(text) > 200 else ''}")

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