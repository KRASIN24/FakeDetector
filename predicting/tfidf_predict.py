from models.model_utils import load_model

def predict_articles(articles):
    model, vectorizer = load_model()
    X_new = vectorizer.transform(articles)
    predictions = model.predict(X_new)
    return predictions

