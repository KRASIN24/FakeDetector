from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from processing import preprocess

def main():

    # 1) Load data
    df = preprocess.load_processed('data/processed/news.csv', combine_title=True)

    # 3) Prepare TF-IDF train/test
    X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer = preprocess.prepare_tfidf(df)

    # 4) Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # 5) Evaluate
    y_pred = model.predict(X_test_tfidf)
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # 6) Nice confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Fake','True'], yticklabels=['Fake','True'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == '__main__':
    main()