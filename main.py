"""
Analyst Agent - RSS -> Deduplicate -> Groq -> CSV
Run in terminal: python main.py
"""

from __future__ import annotations
import os
import time
import json
import logging
import re
import html
from typing import List, Dict, Any, Tuple

import feedparser
import requests
import pandas as pd
from dotenv import load_dotenv
import difflib
from difflib import SequenceMatcher

# Load .env
load_dotenv()

# ---------- Config ----------
RSS_FEEDS: List[str] = [
    "https://techcrunch.com/tag/artificial-intelligence/feed/",
    "https://thenextweb.com/feed/"
]
MAX_ARTICLES = 30
SIMILARITY_THRESHOLD = 75
MIN_CONTENT_WORDS = 60
PROMO_KEYWORDS = [
    "game-changer", "revolutionary", "best-in-class", "industry-leading",
    "breakthrough", "innovative", "cutting-edge"
]
OUTPUT_DIR = "outputs"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "clean_articles.csv")
GROQ_API_KEY = os.getenv("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")
GROQ_API_URL = os.getenv("https://api.groq.com/openai/v1/chat/completions")  # e.g. https://api.groq.ai/v1/generate
REQUEST_TIMEOUT = 20
# ----------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def safe_text(s: str) -> str:
    if not s:
        return ""
    s = html.unescape(s)
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_articles(feeds: List[str], max_articles: int = MAX_ARTICLES) -> List[Dict[str, Any]]:
    articles: List[Dict[str, Any]] = []
    for feed_url in feeds:
        logging.info("Fetching feed: %s", feed_url)
        parsed = feedparser.parse(feed_url)
        for entry in parsed.entries:
            if len(articles) >= max_articles:
                break
            title = entry.get("title", "") or ""
            link = entry.get("link", "") or ""
            content = ""
            if "content" in entry and entry.content:
                content = entry.content[0].value
            else:
                content = entry.get("summary", "") or ""
            content = safe_text(content)
            published = entry.get("published", entry.get("updated", ""))
            articles.append({
                "title": title,
                "link": link,
                "content": content,
                "published": published,
                "source_feed": feed_url
            })
    logging.info("Fetched %d articles", len(articles))
    return articles


def deduplicate_by_title(articles, threshold=0.75):
    unique = []
    seen_titles = []

    for art in articles:
        title = (art.get("title") or "").strip().lower()
        if not title:
            continue

        is_dup = False
        for seen in seen_titles:
            similarity = difflib.SequenceMatcher(None, title, seen).ratio()
            if similarity >= threshold:
                is_dup = True
                break

        if not is_dup:
            seen_titles.append(title)
            unique.append(art)

    return unique



def hype_filter(article: Dict[str, Any]) -> Tuple[bool, str]:
    content = article.get("content", "") or ""
    word_count = len(content.split())
    if word_count < MIN_CONTENT_WORDS:
        return False, f"too_short ({word_count})"
    lower = content.lower()
    promo_hits = sum(1 for k in PROMO_KEYWORDS if k in lower)
    title = (article.get("title") or "").lower()
    title_promo_hits = sum(1 for k in PROMO_KEYWORDS if k in title)
    promo_density = promo_hits / max(1, word_count)
    if promo_density > 0.01 or title_promo_hits >= 1:
        return False, f"promo ({promo_hits}, density={promo_density:.3f})"
    return True, "ok"


def build_groq_prompt(article: Dict[str, Any]) -> str:
    title = article.get("title", "")
    content = article.get("content", "")
    prompt = (
        "You are a JSON extractor. Given TITLE and CONTENT, return ONLY a JSON object:\n\n"
        '{ "company_name": "", "category": "", "sentiment_score": 0.0, "is_funding_news": false }\n\n'
        "TITLE: " + title + "\n\n"
        "CONTENT: " + content[:3000] + "\n\n"
        "Rules: keep values simple. Use empty string if unknown. Category short (e.g., 'Gen AI')."
    )
    return prompt


def call_groq_extract(article: Dict[str, Any]) -> Dict[str, Any]:
    if not GROQ_API_KEY or not GROQ_API_URL:
        raise RuntimeError("Set GROQ_API_KEY and GROQ_API_URL in .env")

    prompt = build_groq_prompt(article)

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "mixtral-8x7b-32768",   # fastest + best for extraction
        "messages": [
            {"role": "system", "content": "You extract structured JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 200
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    data = response.json()

    # Extract LLM message text
    try:
        model_text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        print("Groq Response:", data)
        return {"raw": data}

    # Parse JSON from the LLM output
    try:
        return json.loads(model_text)
    except:
        return {"raw": model_text}
89

def run_pipeline() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    articles = fetch_articles(RSS_FEEDS, max_articles=MAX_ARTICLES)
    if not articles:
        logging.error("No articles fetched, exiting.")
        return
    deduped = deduplicate_by_title(articles)
    results: List[Dict[str, Any]] = []
    for art in deduped:
        keep, reason = hype_filter(art)
        if not keep:
            logging.info("Filtered: %s -> %s", (art.get("title") or "")[:80], reason)
            continue
        try:
            extracted = call_groq_extract(art)
        except Exception as exc:
            logging.warning("LLM extraction failed: %s", exc)
            extracted = {"company_name": "", "category": "Unknown", "sentiment_score": 0.0, "is_funding_news": False, "error": str(exc)}
        row = {
            "title": art.get("title"),
            "link": art.get("link"),
            "published": art.get("published"),
            "source_feed": art.get("source_feed"),
            "company_name": extracted.get("company_name", ""),
            "category": extracted.get("category", "Unknown"),
            "sentiment_score": extracted.get("sentiment_score", 0.0),
            "is_funding_news": extracted.get("is_funding_news", False),
            "raw_extraction": json.dumps(extracted, ensure_ascii=False)
        }
        results.append(row)
        time.sleep(0.25)
    if not results:
        logging.warning("No results to save.")
        return
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)
    logging.info("Saved %d rows to %s", len(df), OUTPUT_CSV)
    print(f"Saved {len(df)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    run_pipeline()
