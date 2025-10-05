import joblib
import os

def save_model(model, vectorizer, folder='models'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    joblib.dump(model, f'{folder}/fake_news_model.pkl')
    joblib.dump(vectorizer, f'{folder}/tfidf_vectorizer.pkl')
    print("Model and vectorizer saved!")

def load_model(folder='models'):
    model = joblib.load(f'{folder}/fake_news_model.pkl')
    vectorizer = joblib.load(f'{folder}/tfidf_vectorizer.pkl')
    return model, vectorizer