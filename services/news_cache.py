import hashlib
import os
import json
from datetime import datetime

CACHE_DIR = "data/cache/news"
os.makedirs(CACHE_DIR, exist_ok=True)

def _cache_path(dataset: str, source: str, query: str, language: str, limit: int) -> str:
    raw_key = f"{source}|{query}|{language}|{limit}"
    key = hashlib.md5(raw_key.encode("utf-8")).hexdigest()

    dataset_dir = os.path.join(CACHE_DIR, dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    return os.path.join(dataset_dir, f"{key}.json")

def load_cache(source: str, query: str):
    path = _cache_path(source, query)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "..", "data", "cache", "news")

def load_cached_articles(dataset: str):
    path = os.path.join(CACHE_DIR, f"{dataset}.json")

    # print("Looking for dataset at:", path)  # optional debug

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset '{dataset}' does not exist")

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    return payload["articles"]

def save_cache(source: str, query: str, articles: list):
    path = _cache_path(source, query)
    payload = {
        "source": source,
        "query": query,
        "timestamp": datetime.utcnow().isoformat(),
        "count": len(articles),
        "articles": articles,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
