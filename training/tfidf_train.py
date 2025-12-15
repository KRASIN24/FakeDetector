import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from processing import preprocess
from models.model_utils import save_model
from predicting.tfidf_predict import predict_articles


def train_tfidf_model():
    # 1) Load data
    df = preprocess.load_processed(
        "data/processed/news.csv",
        combine_title=True
    )

    # 2) Prepare TF-IDF features
    X_train, X_test, y_train, y_test, vectorizer = preprocess.prepare_tfidf(df)

    # 3) Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 4) Evaluate
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # 5) Save model & vectorizer
    model_folder = "models"
    os.makedirs(model_folder, exist_ok=True)

    model_path = os.path.join(model_folder, "fake_news_model.pkl")
    vectorizer_path = os.path.join(model_folder, "tfidf_vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        save_model(model, vectorizer)
    else:
        print("Model and vectorizer already exist. Skipping save.")

    # 6) Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Fake", "True"],
        yticklabels=["Fake", "True"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("TF-IDF Confusion Matrix")
    # plt.show()

    # # 7) Example inference
    # new_articles = [
    #     "Breaking news: something happened in politics today...",
    #     "Celebrity scandal goes viral online!"
    # ]
    #
    # preds = predict_articles(new_articles)
    # for article, pred in zip(new_articles, preds):
    #     label = "True" if pred == 1 else "Fake"
    #     print(f"{label}: {article}")
