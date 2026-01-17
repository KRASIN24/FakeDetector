import os
import pickle
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from processing import preprocess
from models.auxiliary_features import AuxiliaryFeatureExtractor


def train_tfidf_model(
        data_path="data/processed/news.csv",
        save_dir="models/tfidf",
        aux_features=None,
        test_size=0.2,
        random_state=42
):
    """
    Train TF-IDF + Logistic Regression model

    Args:
        data_path: Path to training data CSV
        save_dir: Directory to save model (models/tfidf or models/tfidf_aux)
        aux_features: If True, compute and use auxiliary features
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    print(f"\n{'=' * 80}")
    print(f"Training TF-IDF Model {'WITH' if aux_features else 'WITHOUT'} Auxiliary Features")
    print(f"{'=' * 80}\n")

    # 1) Load data
    print("📂 Loading data...")
    df = preprocess.load_processed(data_path, combine_title=True)
    print(f"   Loaded {len(df)} samples")

    # 2) Prepare TF-IDF features
    print("\n🔧 Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.8
    )

    X_tfidf = vectorizer.fit_transform(df['text'])
    y = df['label'].values
    print(f"   TF-IDF shape: {X_tfidf.shape}")

    # 3) Optionally add auxiliary features
    X_aux_data = None
    if aux_features:
        print("\n🔧 Computing auxiliary features (stance + emotion)...")

        # Use batch processing with progress bar
        aux_extractor = AuxiliaryFeatureExtractor(stance=True, emotion=True, batch_size=32)
        X_aux_df = aux_extractor.compute_aux_features(df['text'].tolist(), show_progress=True)
        X_aux_data = X_aux_df.values
        print(f"   ✓ Aux features shape: {X_aux_data.shape}")

        # Combine TF-IDF + aux
        print("   Combining TF-IDF + aux features...")
        X_combined = hstack([X_tfidf, X_aux_data])
        print(f"   Combined shape: {X_combined.shape}")
    else:
        X_combined = X_tfidf

    # 4) Train/test split
    print(f"\n✂️  Splitting data (test_size={test_size})...")
    if aux_features:
        X_train, X_test, y_train, y_test, X_aux_train, X_aux_test = train_test_split(
            X_combined, y, X_aux_data,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        X_aux_train = None
        X_aux_test = None

    print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # 5) Train model
    print("\n🏋️ Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_train, y_train)

    # 6) Evaluate
    print("\n📊 Evaluating on test set...")
    y_pred = model.predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "True"]))

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    # 7) Save model & vectorizer
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, "model.pkl")
    vectorizer_path = os.path.join(save_dir, "vectorizer.pkl")
    config_path = os.path.join(save_dir, "config.pkl")

    print(f"\n💾 Saving model to {save_dir}...")

    # Save model
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"   ✓ Model saved to {model_path}")

    # Save vectorizer
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"   ✓ Vectorizer saved to {vectorizer_path}")

    # Save config (metadata about aux features)
    config = {
        "use_aux": bool(aux_features),  # Fix: properly check boolean value
        "aux_dim": X_aux_data.shape[1] if X_aux_data is not None else 0,
        "tfidf_features": X_tfidf.shape[1],
        "accuracy": acc,
    }
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"   ✓ Config saved to {config_path}")

    # 8) Confusion matrix
    print("\n📈 Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "True"],
        yticklabels=["Fake", "True"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"TF-IDF Confusion Matrix {'(with aux)' if aux_features else ''}")

    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Confusion matrix saved to {cm_path}")
    plt.close()

    print(f"\n{'=' * 80}")
    print("✅ Training completed successfully!")
    print(f"{'=' * 80}\n")

    return model, vectorizer