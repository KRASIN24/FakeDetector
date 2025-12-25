from __future__ import annotations

import os
import time
import requests
from typing import List, Dict, Literal, Optional

from newsapi import NewsApiClient

SourceType = Literal["newsapi", "gnews"]




class UnifiedNewsClient:
    """
    Unified client for fetching news articles from:
    - NewsAPI.ai (high-credibility / mainstream sources)
    - GNews (mixed-credibility / noisy sources)

    This class is intentionally source-aware to allow
    downstream robustness and domain-shift evaluation.
    """

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        gnews_key: Optional[str] = None,
        request_delay: float = 1.0,
    ):
        self.newsapi_key = newsapi_key or os.getenv("NEWSAPI_KEY")
        self.gnews_key = gnews_key or os.getenv("GNEWS_API_KEY")
        self.request_delay = request_delay

        if not self.newsapi_key and not self.gnews_key:
            raise ValueError("At least one API key must be provided")

        self._newsapi_client = None
        if self.newsapi_key:
            self._newsapi_client = NewsApiClient(api_key=self.newsapi_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(
        self,
        query: str,
        source: SourceType,
        language: str = "en",
        limit: int = 100,
    ) -> List[Dict]:
        """
        Fetch articles from a selected source.

        Args:
            query: Search query
            source: "newsapi" | "gnews"
            language: ISO language code
            limit: Maximum number of articles

        Returns:
            List of normalized article dictionaries
        """
        if source == "newsapi":
            return self._fetch_newsapi(query, language, limit)
        elif source == "gnews":
            return self._fetch_gnews(query, language, limit)
        else:
            raise ValueError(f"Unsupported source: {source}")

    # ------------------------------------------------------------------
    # NewsAPI.ai implementation
    # ------------------------------------------------------------------

    def _fetch_newsapi(
        self,
        query: str,
        language: str,
        limit: int,
    ) -> List[Dict]:
        if not self._newsapi_client:
            raise RuntimeError("NewsAPI client not initialized")

        response = self._newsapi_client.get_everything(
            q=query,
            language=language,
            page_size=min(limit, 100),
            sort_by="publishedAt",
        )

        articles = []
        for item in response.get("articles", []):
            articles.append(
                {
                    "source": "newsapi",
                    "title": item.get("title"),
                    "text": item.get("content") or item.get("description"),
                    "url": item.get("url"),
                    "published_at": item.get("publishedAt"),
                }
            )

        time.sleep(self.request_delay)
        return articles

    # ------------------------------------------------------------------
    # GNews implementation
    # ------------------------------------------------------------------

    def _fetch_gnews(
        self,
        query: str,
        language: str,
        limit: int,
    ) -> List[Dict]:
        if not self.gnews_key:
            raise RuntimeError("GNews API key not provided")

        url = "https://gnews.io/api/v4/search"
        params = {
            "q": query,
            "lang": language,
            "max": min(limit, 100),
            "token": self.gnews_key,
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        articles = []
        for item in data.get("articles", []):
            articles.append(
                {
                    "source": "gnews",
                    "title": item.get("title"),
                    "text": item.get("content"),
                    "url": item.get("url"),
                    "published_at": item.get("publishedAt"),
                }
            )

        time.sleep(self.request_delay)
        return articles
