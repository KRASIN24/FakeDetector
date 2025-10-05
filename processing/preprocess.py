import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def merge_and_save(true_path, fake_path, out_path='data/processed/news.csv', random_state=42):
    """Read true/fake CSVs, add labels, shuffle and save merged file."""
    true_df = pd.read_csv(true_path)
    fake_df = pd.read_csv(fake_path)
    true_df['label'] = 1
    fake_df['label'] = 0
    df = pd.concat([true_df, fake_df], axis=0)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    return df

def load_processed(path='data/processed/news.csv', combine_title=True):
    """Load processed CSV and (optionally) combine title + text into 'text' column."""
    df = pd.read_csv(path)
    if combine_title and 'title' in df.columns and 'text' in df.columns:
        df['text'] = df['title'].astype(str) + " " + df['text'].astype(str)
    # ensure text column exists
    if 'text' not in df.columns:
        raise ValueError("No 'text' column found in processed file.")
    return df

def prepare_tfidf(df, text_col='text', label_col='label',
                  max_features=10000, test_size=0.2, random_state=42, stop_words='english', stratify=True):
    """Split and vectorize text using TF-IDF. Returns X_train, X_test, y_train, y_test, vectorizer."""
    X = df[text_col].fillna('').astype(str)
    y = df[label_col]
    strat = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer