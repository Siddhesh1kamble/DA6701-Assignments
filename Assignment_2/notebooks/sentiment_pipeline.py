"""
sentiment_pipeline.py  —  FinBERT News Sentiment Pipeline
===========================================================
PART A  Fetch Google News headlines per stock (RSS + feedparser)
PART B  Run FinBERT sentiment inference in batches
PART C  Aggregate daily, compute rolling features, merge into datasets

News cache:  data/news/{STOCK}_news_raw.csv
Scored news: data/news/{STOCK}_news_scored.csv
Daily sent:  data/news/{STOCK}_daily_sentiment.csv
Final out:   data/processed/{STOCK}_Dataset.csv (overwritten)

Usage:
    python notebooks/sentiment_pipeline.py
"""

import os
import re
import time
import json
import glob
import warnings
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from GoogleNews import GoogleNews
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)  # suppress HuggingFace info logs

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
NEWS_DIR      = os.path.join(BASE_DIR, "data", "news")
os.makedirs(NEWS_DIR, exist_ok=True)

# Date range — END_DATE is today so recently fetched articles aren't filtered out
START_DATE = datetime(2020, 1, 1)
END_DATE   = datetime.today().replace(hour=23, minute=59, second=59, microsecond=0)

# FinBERT batch size (lower = less RAM; higher = faster if GPU available)
BATCH_SIZE = 32
# Max headlines per RSS call
RSS_MAX    = 100
# Delay between GoogleNews requests (seconds) — avoids 429 rate-limiting
RSS_DELAY  = 3.5

# Required sentiment columns in final dataset
SENTIMENT_COLS = [
    "news_count",
    "sentiment_mean",
    "sentiment_std",
    "sentiment_positive_ratio",
    "sentiment_negative_ratio",
    "sentiment_3d_mean",
    "sentiment_7d_mean",
    "sentiment_30d_mean",
]


# ═════════════════════════════════════════════════════════════════════════════
# PART A — NEWS COLLECTION
# ═════════════════════════════════════════════════════════════════════════════

def build_query_map() -> dict[str, list[str]]:
    """
    Per-stock search queries.
    Multiple queries improve recall; duplicates are removed later.
    """
    return {
        "BHARTIARTL": [
            "Bharti Airtel stock",
            "BHARTIARTL NSE",
            "Airtel India earnings",
            "Bharti Airtel results",
        ],
        "HDFCBANK": [
            "HDFC Bank stock",
            "HDFCBANK NSE",
            "HDFC Bank results",
            "HDFC Bank earnings",
        ],
        "HINDUNILVR": [
            "Hindustan Unilever stock",
            "HINDUNILVR NSE",
            "HUL earnings",
            "HUL results quarterly",
        ],
        "INFY": [
            "Infosys stock",
            "INFY NSE",
            "Infosys earnings",
            "Infosys quarterly results",
        ],
        "M&M": [
            "Mahindra Mahindra stock",
            "M&M NSE India",
            "Mahindra quarterly results",
            "Mahindra auto earnings",
        ],
        "RELIANCE": [
            "Reliance Industries stock",
            "RELIANCE NSE",
            "RIL earnings",
            "Reliance Industries results",
        ],
    }


def _clean_headline(text: str) -> str:
    """Remove HTML entities, extra whitespace, and non-ASCII noise."""
    if not text:
        return ""
    # Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = (text.replace("&amp;", "&").replace("&lt;", "<")
                .replace("&gt;", ">").replace("&quot;", '"')
                .replace("&#39;", "'").replace("&nbsp;", " "))
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fetch_gnews_chunk(
    query: str,
    start: datetime,
    end: datetime,
) -> list[dict]:
    """
    Fetch headlines for a single (query, month) chunk via GoogleNews library.
    Returns list of {date, headline, source}.
    """
    start_str = start.strftime("%m/%d/%Y")
    end_str   = end.strftime("%m/%d/%Y")
    try:
        gn = GoogleNews(lang='en', region='IN',
                        start=start_str, end=end_str,
                        encode='utf-8')
        gn.search(query)
        results = gn.results()
        rows = []
        for r in results:
            raw_date = r.get('date', '')
            headline = _clean_headline(r.get('title', ''))
            source   = r.get('media', '')
            if not headline:
                continue
            # Parse the date string (GoogleNews returns relative or absolute dates)
            parsed_date = _parse_gnews_date(raw_date, end)
            rows.append({'date': parsed_date, 'headline': headline, 'source': source})
        return rows
    except Exception as exc:
        print(f"    [WARN] GoogleNews fetch failed ('{query}' {start_str}-{end_str}): {exc}")
        return []


def _parse_gnews_date(date_str: str, ref_date: datetime) -> str | None:
    """
    Convert GoogleNews date string to YYYY-MM-DD.
    Handles both relative ('3 hours ago', '2 days ago') and
    absolute ('Feb 20, 2025') formats.
    """
    if not date_str:
        return None
    s = date_str.strip().lower()
    # Relative: 'X minutes/hours ago' -> treat as ref_date
    if 'ago' in s or 'minute' in s or 'hour' in s or 'just now' in s:
        return ref_date.strftime('%Y-%m-%d')
    # Relative: 'X days/weeks ago'
    try:
        import re as _re
        m = _re.search(r'(\d+)\s*(day|week)', s)
        if m:
            n = int(m.group(1))
            delta = timedelta(days=n) if 'day' in m.group(2) else timedelta(weeks=n)
            return (ref_date - delta).strftime('%Y-%m-%d')
    except Exception:
        pass
    # Absolute: try multiple formats
    for fmt in ('%b %d, %Y', '%B %d, %Y', '%d %b %Y', '%Y-%m-%d', '%m/%d/%Y'):
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime('%Y-%m-%d')
        except ValueError:
            continue
    return None


def _is_financial_noise(headline: str) -> bool:
    """
    Simple heuristic filter for obviously non-financial content.
    Returns True if headline looks like noise.
    """
    noise_patterns = [
        r"^(sports|cricket|ipl|football|entertainment|bollywood)",
        r"\b(recipe|weather|horoscope|astrology|celebrity)\b",
    ]
    h = headline.lower()
    return any(re.search(p, h) for p in noise_patterns)


def fetch_stock_news(
    stock: str,
    queries: list[str],
    cache_path: str,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch, deduplicate, and cache Google News headlines for one stock.

    Uses cache (data/news/{STOCK}_news_raw.csv) to avoid repeated fetches.
    Appends newly fetched articles to the cache on each run.

    Returns a DataFrame with columns: date, headline, source
    """
    # ── Load existing cache ─────────────────────────────────────────────────
    if os.path.exists(cache_path) and not force_refresh:
        cached = pd.read_csv(cache_path, parse_dates=["date"])
        print(f"  [Cache] Loaded {len(cached)} cached headlines for {stock}")
    else:
        cached = pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                                "headline": pd.Series(dtype=str),
                                "source": pd.Series(dtype=str)})

    # -- Fetch headlines: one gn.search() per query, 8s inter-query delay --
    new_rows = []
    today = datetime.today()
    for q in queries:
        for attempt in range(3):
            try:
                gn = GoogleNews(lang='en', region='IN', encode='utf-8')
                gn.search(q)
                for r in gn.results():
                    hl = _clean_headline(r.get('title', ''))
                    if not hl:
                        continue
                    pdate = _parse_gnews_date(r.get('date', ''), today)
                    new_rows.append({'date': pdate, 'headline': hl,
                                     'source': r.get('media', '')})
                gn.clear()
                break  # success
            except Exception as exc:
                wait = (2 ** attempt) * 10
                print(f"    [WARN] '{q}' attempt {attempt+1}: {exc} -> wait {wait}s")
                time.sleep(wait)
        time.sleep(8)  # 8s cooldown between each query

    # ── Combine, deduplicate, filter ────────────────────────────────────────
    if new_rows:
        fresh = pd.DataFrame(new_rows, columns=["date", "headline", "source"])
        fresh["date"] = pd.to_datetime(fresh["date"], errors="coerce")
        combined = pd.concat([cached, fresh], ignore_index=True)
    else:
        combined = cached.copy()

    # Ensure required columns always present
    for _c in ["date", "headline", "source"]:
        if _c not in combined.columns:
            combined[_c] = np.nan
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")

    # Deduplicate by (date, headline) after normalizing text case
    combined["_key"] = combined["headline"].str.lower().str.strip()
    combined = combined.drop_duplicates(subset=["date", "_key"]).drop(columns=["_key"])

    # Drop rows with no date or empty headline
    combined = combined.dropna(subset=["date", "headline"])
    combined = combined[combined["headline"].str.strip() != ""]

    # Filter date range (safe on empty frames)
    if len(combined) > 0:
        combined = combined[
            (combined["date"] >= START_DATE) & (combined["date"] <= END_DATE)
        ].copy()

    # Drop financial noise (safe on empty frames)
    if len(combined) > 0:
        combined = combined[~combined["headline"].apply(_is_financial_noise)].copy()

    # Sort (always safe)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Save updated cache
    combined.to_csv(cache_path, index=False)
    n_dates = int(combined["date"].nunique()) if len(combined) > 0 else 0
    print(f"  [News] {stock}: {len(combined)} unique headlines "
          f"({n_dates} distinct dates)")

    return combined


# ═════════════════════════════════════════════════════════════════════════════
# PART B — FINBERT SENTIMENT INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def load_finbert():
    """
    Load ProsusAI/finbert tokenizer and model.

    Auto-downloads model weights (~440 MB) on first run.
    Uses GPU if available, falls back to CPU.
    Returns (pipeline_object, device_str).
    """
    try:
        import torch
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        device_name = "cuda" if device == 0 else "cpu"
        print(f"  [FinBERT] Loading ProsusAI/finbert on {device_name}...")

        sentiment_pipe = hf_pipeline(
            task="text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=device,
            top_k=None,          # return all 3 class probabilities
            truncation=True,
            max_length=512,
        )
        print(f"  [FinBERT] Model ready.")
        return sentiment_pipe, device_name

    except ImportError as e:
        raise RuntimeError(
            f"torch/transformers not importable: {e}\n"
            "Run: pip install torch transformers"
        )


def _score_batch(headlines: list[str], pipe) -> list[dict]:
    """
    Run FinBERT on a batch of headlines.
    Returns list of {positive, negative, neutral, sentiment_score, label}.
    """
    if not headlines:
        return []

    results = pipe(headlines, batch_size=BATCH_SIZE, truncation=True)
    scored = []
    for res in results:
        probs = {item["label"].lower(): item["score"] for item in res}
        pos   = probs.get("positive", 0.0)
        neg   = probs.get("negative", 0.0)
        neu   = probs.get("neutral",  0.0)
        score = pos - neg  # range [-1, +1]
        label = max(probs, key=probs.get)
        scored.append({
            "positive_prob": round(pos, 4),
            "negative_prob": round(neg, 4),
            "neutral_prob":  round(neu, 4),
            "sentiment_score": round(score, 4),
            "predicted_label": label,
        })
    return scored


def score_news(
    news_df: pd.DataFrame,
    scored_cache_path: str,
    pipe,
) -> pd.DataFrame:
    """
    Score each headline with FinBERT; cache results.

    If a scored cache exists, only score new (unseen) headlines.

    Returns scored DataFrame with all original columns + sentiment columns.
    """
    # Load existing scored cache
    if os.path.exists(scored_cache_path):
        scored_cache = pd.read_csv(scored_cache_path)
        already_scored = set(scored_cache["headline"].str.lower().str.strip())
        print(f"  [Score] Loaded {len(scored_cache)} cached scored headlines")
    else:
        scored_cache  = pd.DataFrame()
        already_scored = set()

    # Find unscored headlines
    news_df["_norm"] = news_df["headline"].str.lower().str.strip()
    unscored = news_df[~news_df["_norm"].isin(already_scored)].copy()
    news_df  = news_df.drop(columns=["_norm"])

    if len(unscored) > 0:
        print(f"  [Score] Scoring {len(unscored)} new headlines in batches of {BATCH_SIZE}...")
        headlines = unscored["headline"].tolist()

        all_scores = []
        for i in range(0, len(headlines), BATCH_SIZE):
            batch = headlines[i: i + BATCH_SIZE]
            all_scores.extend(_score_batch(batch, pipe))
            if (i // BATCH_SIZE) % 5 == 0:
                print(f"    Batch {i // BATCH_SIZE + 1}/{(len(headlines) + BATCH_SIZE - 1) // BATCH_SIZE}...")

        scores_df = pd.DataFrame(all_scores)
        unscored = unscored.reset_index(drop=True)
        unscored = pd.concat([unscored.drop(columns=["_norm"], errors="ignore"), scores_df], axis=1)

        # Append to cache
        full_scored = pd.concat([scored_cache, unscored], ignore_index=True)
        full_scored.to_csv(scored_cache_path, index=False)
        print(f"  [Score] Total cached scored headlines: {len(full_scored)}")
    else:
        full_scored = scored_cache
        print(f"  [Score] All headlines already scored (cache hit)")

    return full_scored


# ═════════════════════════════════════════════════════════════════════════════
# PART C — DAILY AGGREGATION & MERGE
# ═════════════════════════════════════════════════════════════════════════════

def aggregate_daily(scored_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-headline scores to daily sentiment features.

    Per day (Date):
      - news_count          : number of headlines
      - sentiment_mean      : mean(sentiment_score)
      - sentiment_std       : std(sentiment_score)
      - sentiment_positive_ratio : frac of headlines labelled positive
      - sentiment_negative_ratio : frac of headlines labelled negative

    Days with no news are NOT padded here (handled at merge stage).
    B4: No forward fill — each day only gets same-day news.
    """
    if "date" in scored_df.columns:
        scored_df = scored_df.rename(columns={"date": "Date"})

    scored_df["Date"] = pd.to_datetime(scored_df["Date"])

    # Guard: only rows with a valid sentiment_score
    scored_df = scored_df.dropna(subset=["sentiment_score"])

    daily = scored_df.groupby("Date").agg(
        news_count=("headline", "count"),
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_std=("sentiment_score", "std"),
        _pos_count=("predicted_label", lambda x: (x == "positive").sum()),
        _neg_count=("predicted_label", lambda x: (x == "negative").sum()),
    ).reset_index()

    daily["sentiment_positive_ratio"] = (
        daily["_pos_count"] / daily["news_count"]
    ).fillna(0.0)
    daily["sentiment_negative_ratio"] = (
        daily["_neg_count"] / daily["news_count"]
    ).fillna(0.0)
    daily = daily.drop(columns=["_pos_count", "_neg_count"])

    # ── C3: Rolling sentiment features (B4 compliant: only past values) ────
    daily = daily.sort_values("Date").reset_index(drop=True)
    daily["sentiment_3d_mean"]  = daily["sentiment_mean"].rolling(3,  min_periods=1).mean()
    daily["sentiment_7d_mean"]  = daily["sentiment_mean"].rolling(7,  min_periods=1).mean()
    daily["sentiment_30d_mean"] = daily["sentiment_mean"].rolling(30, min_periods=1).mean()

    # Clamp sentiment_mean and rolling features to [-1, 1]
    for col in ["sentiment_mean", "sentiment_3d_mean", "sentiment_7d_mean", "sentiment_30d_mean"]:
        daily[col] = daily[col].clip(-1.0, 1.0)

    return daily


def merge_sentiment(
    market_df: pd.DataFrame,
    daily_sentiment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join daily sentiment onto market_df by Date.
    Days with no news get NaN for all sentiment cols EXCEPT news_count (=0).
    C2: No forward fill for sentiment values.
    """
    merged = market_df.merge(daily_sentiment, on="Date", how="left")

    # Fill news_count with 0 where there was no news (not NaN)
    merged["news_count"] = merged["news_count"].fillna(0).astype(int)

    # Enforce float64 for all sentiment cols (except news_count)
    float_sent_cols = [c for c in SENTIMENT_COLS if c != "news_count"]
    for col in float_sent_cols:
        if col not in merged.columns:
            merged[col] = np.nan
        merged[col] = merged[col].astype("float64")

    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Validation
# ═════════════════════════════════════════════════════════════════════════════

def validate_sentiment(df: pd.DataFrame, stock: str, original_rows: int) -> dict:
    """Assert key correctness guarantees; return validation record."""
    # Row count unchanged
    assert len(df) == original_rows, (
        f"[{stock}] Row count changed: {original_rows} -> {len(df)}"
    )
    # All sentiment columns exist
    missing = [c for c in SENTIMENT_COLS if c not in df.columns]
    assert not missing, f"[{stock}] Missing sentiment cols: {missing}"

    # Sentiment mean in [-1, 1] (ignore NaN)
    valid = df["sentiment_mean"].dropna()
    if len(valid) > 0:
        assert valid.between(-1, 1).all(), f"[{stock}] sentiment_mean out of [-1,1]"

    # No inf
    for col in SENTIMENT_COLS:
        n_inf = np.isinf(df[col].replace(np.nan, 0)).sum()
        assert n_inf == 0, f"[{stock}].{col}: {n_inf} inf values"

    coverage = df["news_count"].gt(0).mean() * 100
    return {
        "rows":       len(df),
        "coverage_pct": round(float(coverage), 2),
        "total_news": int(df["news_count"].sum()),
        "nan_pct": {
            col: round(float(df[col].isna().mean() * 100), 1)
            for col in SENTIMENT_COLS if col != "news_count"
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("FinBERT News Sentiment Pipeline")
    print("=" * 65)

    query_map = build_query_map()
    files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_Dataset.csv")))
    stocks = [os.path.basename(f).replace("_Dataset.csv", "") for f in files]
    print(f"Stocks: {', '.join(stocks)}")

    # ── Load FinBERT once ─────────────────────────────────────────────────
    print("\n[PART B] Loading FinBERT model...")
    pipe, device = load_finbert()

    coverage_log = {}
    validation_log = {}

    for filepath, stock in zip(files, stocks):
        print(f"\n{'-'*50}")
        print(f"[{stock}]")

        news_cache_path   = os.path.join(NEWS_DIR, f"{stock}_news_raw.csv")
        scored_cache_path = os.path.join(NEWS_DIR, f"{stock}_news_scored.csv")
        daily_sent_path   = os.path.join(NEWS_DIR, f"{stock}_daily_sentiment.csv")

        # ── PART A: Fetch news ────────────────────────────────────────────
        queries = query_map.get(stock, [f"{stock} stock NSE India"])
        print(f"  [PART A] Fetching news ({len(queries)} queries)...")
        news_df = fetch_stock_news(stock, queries, news_cache_path)

        # ── PART B: Score with FinBERT ────────────────────────────────────
        if len(news_df) == 0:
            print(f"  [PART B] No news to score for {stock}")
            scored_df = pd.DataFrame()
        else:
            scored_df = score_news(news_df, scored_cache_path, pipe)

        # ── PART C: Aggregate daily ───────────────────────────────────────
        if len(scored_df) > 0 and "sentiment_score" in scored_df.columns:
            daily_sentiment = aggregate_daily(scored_df)
            daily_sentiment.to_csv(daily_sent_path, index=False)
            print(f"  [PART C] Daily sentiment: {len(daily_sentiment)} rows with news")
        else:
            # No news at all → empty daily sentiment
            daily_sentiment = pd.DataFrame(columns=["Date"] + SENTIMENT_COLS)
            print(f"  [PART C] No scored headlines — all sentiment will be NaN")

        # ── Merge into processed dataset ──────────────────────────────────
        market_df = pd.read_csv(filepath, parse_dates=["Date"])
        original_rows = len(market_df)

        # Remove any existing sentiment cols (idempotent)
        market_df = market_df.drop(columns=SENTIMENT_COLS, errors="ignore")

        merged = merge_sentiment(market_df, daily_sentiment)

        # ── Validate ──────────────────────────────────────────────────────
        val = validate_sentiment(merged, stock, original_rows)
        validation_log[stock] = val
        coverage_log[stock] = val["coverage_pct"]

        print(f"  Coverage: {val['coverage_pct']}% of trading days have news | "
              f"Total headlines used: {val['total_news']}")

        # ── Save updated dataset ──────────────────────────────────────────
        merged.to_csv(filepath, index=False)
        print(f"  [OK] Saved -> {filepath}  ({merged.shape[0]} rows x {merged.shape[1]} cols)")

    # ── Save logs ─────────────────────────────────────────────────────────
    val_log_path = os.path.join(PROCESSED_DIR, "sentiment_validation_log.json")
    with open(val_log_path, "w") as f:
        json.dump(validation_log, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    for stock, cov in coverage_log.items():
        print(f"  {stock:<20}  coverage={cov:.1f}%")

    # Cross-stock schema check
    sent_sets = set()
    for filepath in files:
        cols = pd.read_csv(filepath, nrows=0).columns.tolist()
        sent_sets.add(frozenset(c for c in cols if c in SENTIMENT_COLS))
    if len(sent_sets) == 1:
        print("\n[OK] All stocks have IDENTICAL sentiment schema.")
    else:
        print("\n[WARN] Sentiment schema mismatch!")

    print(f"\nLogs saved: {val_log_path}")
    print("Done.")


if __name__ == "__main__":
    main()
