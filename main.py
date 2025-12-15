from predicting.tfidf_predict import predict_articles
import argparse

from training.tfidf_train import train_tfidf_model


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

    if args.model == "tfidf":
        train_tfidf_model()
    else:
        raise ValueError(f"Unsupported model type: {args.model}")


    # Predicting
    new_articles = [
        "Breaking news: something happened in politics today...",
        "Celebrity scandal goes viral online!"
    ]

    preds = predict_articles(new_articles)
    for article, pred in zip(new_articles, preds):
        label = "True" if pred == 1 else "Fake"
        print(f"{label}: {article}")



if __name__ == "__main__":
    main()


