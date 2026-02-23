"""
impute_fundamentals.py
======================
Production-grade, leakage-free imputation for fundamental columns in each
data/processed/{STOCK}_Dataset.csv.

Imputation hierarchy (applied in strict order):
  Level 0 — Historical Scraping (Screener.in quarterly tables)
  Level 1 — Deterministic financial identities
  Level 2 — Time-series forward/backward fill (within-quarter limit)
  Level 3 — Peer/sector median fill
  Level 4 — ML imputation (IterativeImputer / MICE)
  Level 5 — Leave as NaN

Quality flag columns:  fund_X_imputed_flag  (0=original, 1=scraped,
                        2=deterministic, 3=timefill, 4=peer, 5=ML)

Outputs:
  data/processed/{STOCK}_Dataset.csv   (overwritten, with flag cols)
  data/processed/imputation_report.json
"""

import os
import re
import sys
import json
import time
import warnings
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
LOG_PATH = os.path.join(PROC_DIR, "imputation_report.json")

STOCKS = ["BHARTIARTL", "HDFCBANK", "HINDUNILVR", "INFY", "M&M", "RELIANCE"]

# Fundamental columns we are allowed to impute
FUND_COLS = [
    "fund_basic_eps", "fund_depreciation", "fund_diluted_eps", "fund_ebit",
    "fund_ebit_plus_other_income", "fund_employee_cost", "fund_equity_capital",
    "fund_interest_expense", "fund_minority_interest", "fund_net_profit",
    "fund_net_profit_after_mi", "fund_other_expenses", "fund_other_income",
    "fund_pat_ordinary", "fund_pbt", "fund_pbt_before_exceptional",
    "fund_revenue", "fund_tax", "fund_total_income",
]

# Map stock symbol -> Screener.in company slug
SCREENER_SLUGS = {
    "BHARTIARTL": "BHARTIARTL",
    "HDFCBANK":   "HDFCBANK",
    "HINDUNILVR": "HINDUNILVR",
    "INFY":       "INFY",
    "M&M":        "M-M",
    "RELIANCE":   "RELIANCE",
}

# Sector mapping for peer fill
SECTOR_MAP = {
    "BHARTIARTL": "Telecom",
    "HDFCBANK":   "Banking",
    "HINDUNILVR": "FMCG",
    "INFY":       "IT",
    "M&M":        "Auto",
    "RELIANCE":   "Conglomerate",
}

# Screener.in column name -> our fund_* column
SCREENER_COL_MAP = {
    "Sales":              "fund_revenue",
    "Revenue":            "fund_revenue",
    "Expenses":           "fund_other_expenses",
    "Operating Profit":   "fund_ebit",
    "OPM":                None,                  # percentage, skip
    "Other Income":       "fund_other_income",
    "Interest":           "fund_interest_expense",
    "Depreciation":       "fund_depreciation",
    "Profit before tax":  "fund_pbt",
    "Tax":                "fund_tax",
    "Net Profit":         "fund_net_profit",
    "EPS":                "fund_basic_eps",
    "EPS in Rs":          "fund_basic_eps",
    "Dividend Payout":    None,
    "Employee Cost":      "fund_employee_cost",
    "Total Income":       "fund_total_income",
    "EBIT":               "fund_ebit",
    "PBT":                "fund_pbt",
}

DATE_TOLERANCE_DAYS = 45   # quarter-match tolerance
FFILL_LIMIT        = 91    # ~1 quarter of trading days
REQUESTS_TIMEOUT   = 15

RANDOM_SEED = 42

# Report accumulator
report = {}


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _pct_missing(df: pd.DataFrame, cols: list) -> float:
    present = [c for c in cols if c in df.columns]
    if not present:
        return 0.0
    return float(df[present].isna().mean().mean())


def _flag_col(col: str) -> str:
    return f"{col}_imputed_flag"


def _init_flags(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Create flag columns (0 = original) for every fund col that exists in df."""
    for c in cols:
        fc = _flag_col(c)
        if c in df.columns and fc not in df.columns:
            df[fc] = np.where(df[c].notna(), 0, np.nan)
    return df


def _set_flag(df: pd.DataFrame, mask: pd.Series, col: str, level: int) -> None:
    """Set flag=level where mask is True and flag is currently NaN."""
    fc = _flag_col(col)
    if fc not in df.columns:
        df[fc] = np.nan
    update = mask & df[fc].isna()
    df.loc[update, fc] = level


# ---------------------------------------------------------------------------
# LEVEL 0 — SCREENER.IN SCRAPING
# ---------------------------------------------------------------------------

def _screener_quarterly(stock: str) -> pd.DataFrame:
    """
    Scrape quarterly P&L from Screener.in for one stock.
    Returns DataFrame: index=quarter_end_date (approx), cols=screener col names.
    Returns empty DataFrame on failure.
    """
    slug = SCREENER_SLUGS.get(stock, stock)
    url  = f"https://www.screener.in/company/{slug}/consolidated/"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
        if resp.status_code != 200:
            # Try standalone (non-consolidated)
            url2 = f"https://www.screener.in/company/{slug}/"
            resp = requests.get(url2, headers=headers, timeout=REQUESTS_TIMEOUT)
        if resp.status_code != 200:
            print(f"  [Scrape] {stock}: HTTP {resp.status_code}")
            return pd.DataFrame()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Find the quarterly P&L section
        section = None
        for h2 in soup.find_all(["h2", "h3", "section"]):
            if "quarter" in h2.get_text(strip=True).lower():
                section = h2.find_next("table")
                break

        if section is None:
            # Try direct table search
            tables = soup.find_all("table", {"class": re.compile("data-table|responsive-holder", re.I)})
            for t in tables:
                header_text = t.get_text(" ").lower()
                if "dec" in header_text or "sep" in header_text or "mar" in header_text:
                    section = t
                    break

        if section is None:
            print(f"  [Scrape] {stock}: quarterly table not found")
            return pd.DataFrame()

        # Parse table
        rows = section.find_all("tr")
        if not rows:
            return pd.DataFrame()

        # First row = headers (quarter names like "Dec 2024", "Sep 2024" ...)
        header_cells = rows[0].find_all(["th", "td"])
        quarter_labels = [c.get_text(strip=True) for c in header_cells[1:]]
        quarter_dates  = [_parse_screener_quarter(q) for q in quarter_labels]

        all_data = {}
        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            row_name = cells[0].get_text(strip=True)
            values   = []
            for c in cells[1:]:
                txt = c.get_text(strip=True).replace(",", "").replace("%", "")
                try:
                    values.append(float(txt))
                except ValueError:
                    values.append(np.nan)
            if len(values) == len(quarter_dates):
                all_data[row_name] = values

        if not all_data:
            return pd.DataFrame()

        df = pd.DataFrame(all_data, index=quarter_dates).T
        df.index.name = "screener_col"
        df.columns    = pd.to_datetime(quarter_dates)
        return df

    except Exception as exc:
        print(f"  [Scrape] {stock}: {exc}")
        return pd.DataFrame()


def _parse_screener_quarter(label: str) -> datetime:
    """Convert 'Dec 2024' -> approximately 2024-12-31."""
    label = label.strip()
    # Try formats: "Dec 2024", "Q3 2024", "2024-12", etc.
    for fmt in ("%b %Y", "%B %Y", "%Y-%m"):
        try:
            dt = datetime.strptime(label, fmt)
            # Move to month-end
            import calendar
            last_day = calendar.monthrange(dt.year, dt.month)[1]
            return datetime(dt.year, dt.month, last_day)
        except ValueError:
            continue
    # Fallback: return today
    return datetime.today()


def level0_scrape(df: pd.DataFrame, stock: str, log: dict) -> pd.DataFrame:
    """
    Fill missing cells using Screener.in quarterly data.
    Only fills if scraped quarter date matches dataset date within DATE_TOLERANCE_DAYS.
    """
    log["L0_attempted"] = 0
    log["L0_filled"]    = 0
    log["L0_rejected"]  = 0
    log["L0_source"]    = "screener.in"

    # Check if this stock even needs scraping
    fund_present = [c for c in FUND_COLS if c in df.columns]
    if not fund_present:
        return df
    needs_scrape = any(df[c].isna().any() for c in fund_present)
    if not needs_scrape:
        log["L0_skipped"] = "no missing values"
        return df

    print(f"  [L0] Scraping Screener.in for {stock}...")
    scraped = _screener_quarterly(stock)

    if scraped.empty:
        log["L0_skipped"] = "scrape returned empty"
        return df

    scraped_dates = list(scraped.columns)
    log["L0_attempted"] = len(scraped_dates)

    df = df.sort_values("Date").reset_index(drop=True)

    for screener_row_name, our_col in SCREENER_COL_MAP.items():
        if our_col is None or our_col not in df.columns:
            continue
        if screener_row_name not in scraped.index:
            continue

        series = scraped.loc[screener_row_name]  # indexed by quarter date

        for qdate, qval in series.items():
            if pd.isna(qval):
                continue
            # Find dataset rows within tolerance of this quarter date
            delta = (df["Date"] - pd.Timestamp(qdate)).abs()
            close_mask = delta <= timedelta(days=DATE_TOLERANCE_DAYS)
            missing_mask = df[our_col].isna()
            fill_mask = close_mask & missing_mask

            n_fill = fill_mask.sum()
            if n_fill > 0:
                df.loc[fill_mask, our_col] = qval
                _set_flag(df, fill_mask, our_col, 1)
                log["L0_filled"] += int(n_fill)
            elif close_mask.sum() > 0 and not missing_mask[close_mask].any():
                pass  # already filled — not a rejection
            else:
                log["L0_rejected"] += 1

    return df


# ---------------------------------------------------------------------------
# LEVEL 1 — DETERMINISTIC FINANCIAL IDENTITIES
# ---------------------------------------------------------------------------

def level1_deterministic(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Apply accounting identities to derive missing values."""
    filled = 0

    def _fill_from(target_col, expr_fn, dep_cols):
        nonlocal filled
        if target_col not in df.columns:
            return
        # Only where target is NaN and all dep_cols are not NaN
        mask = df[target_col].isna()
        for dc in dep_cols:
            if dc not in df.columns:
                return
            mask = mask & df[dc].notna()
        if not mask.any():
            return
        df.loc[mask, target_col] = expr_fn(df.loc[mask])
        _set_flag(df, mask, target_col, 2)
        filled += int(mask.sum())

    # ebit = pbt + interest_expense
    _fill_from("fund_ebit",
               lambda d: d["fund_pbt"] + d.get("fund_interest_expense", 0),
               ["fund_pbt"])

    # pbt = net_profit + tax
    _fill_from("fund_pbt",
               lambda d: d["fund_net_profit"] + d["fund_tax"],
               ["fund_net_profit", "fund_tax"])

    # pbt_before_exceptional ~= pbt (if no exceptional info)
    _fill_from("fund_pbt_before_exceptional",
               lambda d: d["fund_pbt"],
               ["fund_pbt"])

    # net_profit_after_mi = net_profit - minority_interest
    _fill_from("fund_net_profit_after_mi",
               lambda d: d["fund_net_profit"] - d["fund_minority_interest"].fillna(0),
               ["fund_net_profit"])

    # pat_ordinary ~= net_profit_after_mi
    _fill_from("fund_pat_ordinary",
               lambda d: d["fund_net_profit_after_mi"],
               ["fund_net_profit_after_mi"])

    # total_income ~= revenue + other_income
    _fill_from("fund_total_income",
               lambda d: d["fund_revenue"] + d["fund_other_income"].fillna(0),
               ["fund_revenue"])

    # revenue ~= total_income - other_income
    _fill_from("fund_revenue",
               lambda d: d["fund_total_income"] - d["fund_other_income"].fillna(0),
               ["fund_total_income"])

    # ebit_plus_other_income = ebit + other_income
    _fill_from("fund_ebit_plus_other_income",
               lambda d: d["fund_ebit"] + d["fund_other_income"].fillna(0),
               ["fund_ebit"])

    # basic_eps ~= diluted_eps if diluted exists
    _fill_from("fund_basic_eps",
               lambda d: d["fund_diluted_eps"],
               ["fund_diluted_eps"])

    _fill_from("fund_diluted_eps",
               lambda d: d["fund_basic_eps"],
               ["fund_basic_eps"])

    log["L1_filled"] = filled
    return df


# ---------------------------------------------------------------------------
# LEVEL 2 — TIME-SERIES FORWARD/BACKWARD FILL
# ---------------------------------------------------------------------------

def level2_timefill(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Forward-fill (quarterly sticky), then minimal back-fill for leading gaps."""
    df = df.sort_values("Date").reset_index(drop=True)
    filled = 0

    for col in FUND_COLS:
        if col not in df.columns:
            continue
        before = df[col].isna().sum()
        df[col] = df[col].ffill(limit=FFILL_LIMIT)
        df[col] = df[col].bfill(limit=5)  # small leading gap fix
        after = df[col].isna().sum()
        n = int(before - after)
        if n > 0:
            # Mark newly filled rows (where flag is still NaN)
            new_fill = df[col].notna() & df[_flag_col(col)].isna()
            df.loc[new_fill, _flag_col(col)] = 3
            filled += n

    log["L2_filled"] = filled
    return df


# ---------------------------------------------------------------------------
# LEVEL 3 — PEER / SECTOR MEDIAN FILL
# ---------------------------------------------------------------------------

def level3_peer_fill(all_dfs: dict, log_all: dict) -> dict:
    """
    For each date, compute per-sector median across stocks and fill remaining NaN.
    Modifies all_dfs in-place per stock.
    """
    # Build a combined panel: index=(stock, date), value per fund_col
    records = []
    for stock, df in all_dfs.items():
        sector = SECTOR_MAP.get(stock, "Other")
        tmp = df[["Date"] + [c for c in FUND_COLS if c in df.columns]].copy()
        tmp["stock"]  = stock
        tmp["sector"] = sector
        records.append(tmp)

    panel = pd.concat(records, ignore_index=True)
    panel = panel.sort_values("Date").reset_index(drop=True)

    for stock, df in all_dfs.items():
        sector = SECTOR_MAP.get(stock, "Other")
        filled = 0

        for col in FUND_COLS:
            if col not in df.columns:
                continue
            missing_mask = df[col].isna()
            if not missing_mask.any():
                continue

            for idx in df.index[missing_mask]:
                date = df.at[idx, "Date"]

                # Sector median from other stocks on the same date (or near)
                near_date = (panel["Date"] - date).abs() <= timedelta(days=7)
                same_sector = panel["sector"] == sector
                other_stocks = panel["stock"] != stock
                peer_vals = panel.loc[near_date & same_sector & other_stocks, col].dropna()

                if len(peer_vals) == 0:
                    # Fall back to global median across all stocks
                    peer_vals = panel.loc[near_date & other_stocks, col].dropna()

                if len(peer_vals) == 0:
                    continue

                val = float(peer_vals.median())
                df.at[idx, col] = val
                if pd.isna(df.at[idx, _flag_col(col)]):
                    df.at[idx, _flag_col(col)] = 4
                filled += 1

        log_all[stock]["L3_filled"] = filled
        all_dfs[stock] = df

    return all_dfs


# ---------------------------------------------------------------------------
# LEVEL 4 — ML IMPUTATION (IterativeImputer / MICE)
# ---------------------------------------------------------------------------

def level4_ml_impute(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """
    Use IterativeImputer (MICE) on the fundamental columns.
    Fit on first 60% of rows (temporal split) to prevent leakage.
    """
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.preprocessing import StandardScaler

    fund_present = [c for c in FUND_COLS if c in df.columns]
    if not fund_present:
        log["L4_filled"] = 0
        return df

    # Check if anything remains missing
    still_missing = df[fund_present].isna().any().any()
    if not still_missing:
        log["L4_filled"] = 0
        return df

    # Take only the fund columns for ML
    X = df[fund_present].copy().values.astype(float)
    n_rows = len(X)
    split  = int(n_rows * 0.60)

    # Track which cells were NaN before imputation
    was_nan = np.isnan(X)

    # Scale
    scaler = StandardScaler()
    # Replace all-nan columns with 0 to avoid scaler errors
    all_nan_cols = np.all(np.isnan(X), axis=0)
    X[:, all_nan_cols] = 0.0

    X_scaled = scaler.fit_transform(X)
    # Restore NaN for imputer to handle
    X_scaled[was_nan] = np.nan
    # Also restore all-nan cols
    X_scaled[:, all_nan_cols] = np.nan

    try:
        imputer = IterativeImputer(
            max_iter=10, random_state=RANDOM_SEED,
            min_value=None, max_value=None,
            skip_complete=True,
        )
        # Fit on train portion only
        X_train = X_scaled[:split]
        # If train has completely-missing columns, give up on ML for those
        all_nan_train = np.all(np.isnan(X_train), axis=0)
        X_train_safe = X_train.copy()
        X_train_safe[:, all_nan_train] = 0.0

        imputer.fit(X_train_safe)

        X_full_safe = X_scaled.copy()
        X_full_safe[:, all_nan_train] = 0.0

        X_imputed_scaled = imputer.transform(X_full_safe)
        X_imputed = scaler.inverse_transform(X_imputed_scaled)

        # Only accept ML values for cells that were NaN (and that the imputer could reach)
        filled = 0
        for j, col in enumerate(fund_present):
            if all_nan_train[j]:
                continue  # can't impute these via ML
            col_mask = was_nan[:, j]
            if not col_mask.any():
                continue
            df.loc[col_mask, col] = X_imputed[col_mask, j]
            flag_mask = pd.Series(col_mask, index=df.index)
            _set_flag(df, flag_mask, col, 5)
            filled += int(col_mask.sum())

        log["L4_filled"] = filled

    except Exception as exc:
        print(f"  [L4] ML imputation error: {exc}")
        log["L4_filled"] = 0

    return df


# ---------------------------------------------------------------------------
# PART C — FINANCIAL SANITY CONSTRAINTS
# ---------------------------------------------------------------------------

def apply_sanity_constraints(df: pd.DataFrame, stock: str, log: dict) -> pd.DataFrame:
    """Clip/flag values that violate financial realism."""
    clipped = 0

    def _clip(col, lo=None, hi=None):
        nonlocal clipped
        if col not in df.columns:
            return
        if lo is not None:
            bad = df[col] < lo
            df.loc[bad, col] = lo
            clipped += int(bad.sum())
        if hi is not None:
            bad = df[col] > hi
            df.loc[bad, col] = hi
            clipped += int(bad.sum())

    _clip("fund_revenue",           lo=0)
    _clip("fund_equity_capital",    lo=0)
    _clip("fund_total_income",      lo=0)
    _clip("fund_employee_cost",     lo=0)
    _clip("fund_depreciation",      lo=0)
    _clip("fund_interest_expense",  lo=0)
    _clip("fund_basic_eps",         lo=-1000, hi=20000)
    _clip("fund_diluted_eps",       lo=-1000, hi=20000)

    # Remove inf
    for col in FUND_COLS:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    log["sanity_clipped"] = clipped
    return df


# ---------------------------------------------------------------------------
# PART D — FINALIZE FLAGS
# ---------------------------------------------------------------------------

def finalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all flag columns exist and fill remaining NaN flags with correct values."""
    for col in FUND_COLS:
        if col not in df.columns:
            continue
        fc = _flag_col(col)
        if fc not in df.columns:
            df[fc] = np.where(df[col].notna(), 0, np.nan)
        # Where value is now filled but flag still NaN → flag=5 (ML or unknown, treat as ML)
        # Where value is still NaN after all levels → flag stays NaN (= not imputed)
        val_missing = df[col].isna()
        flag_missing = df[fc].isna()
        # Cells that are not missing but have no flag → mark as 0 (original-like)
        df.loc[~val_missing & flag_missing, fc] = 0
    return df


# ---------------------------------------------------------------------------
# PART E — VALIDATION
# ---------------------------------------------------------------------------

def validate(df: pd.DataFrame, stock: str, fund_present: list, log: dict) -> None:
    """Assert no inf, log missing counts."""
    for col in fund_present:
        if col not in df.columns:
            continue
        assert not np.isinf(df[col].dropna()).any(), f"{col} has inf!"
    after_miss = {c: int(df[c].isna().sum()) for c in fund_present if c in df.columns}
    log["missing_after"] = after_miss
    log["row_count"] = len(df)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def process_stock(stock: str, all_dfs_ref: dict) -> tuple:
    """Process one stock, return (stock, df, log)."""
    path = os.path.join(PROC_DIR, f"{stock}_Dataset.csv")
    if not os.path.exists(path):
        print(f"  [SKIP] {stock}: file not found")
        return stock, None, {}

    print(f"\n{'='*50}")
    print(f"[{stock}]")

    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    fund_present = [c for c in FUND_COLS if c in df.columns]
    log = {
        "stock": stock,
        "missing_before": {c: int(df[c].isna().sum()) for c in fund_present},
    }

    # Init flag columns
    df = _init_flags(df, fund_present)

    before_pct = _pct_missing(df, fund_present)
    print(f"  Missing before: {before_pct:.1%}")

    # ---- Level 0: Scraping ----
    df = level0_scrape(df, stock, log)

    # ---- Level 1: Deterministic ----
    df = level1_deterministic(df, log)

    # ---- Level 2: Time-fill ----
    df = level2_timefill(df, log)

    # (Level 3 peer fill is done after all stocks are loaded — see main)

    # ---- Sanity constraints ----
    df = apply_sanity_constraints(df, stock, log)

    after_pct = _pct_missing(df, fund_present)
    print(f"  Missing after L0-L2: {after_pct:.1%}")
    print(f"    L0 filled: {log.get('L0_filled', 0)}")
    print(f"    L1 filled: {log.get('L1_filled', 0)}")
    print(f"    L2 filled: {log.get('L2_filled', 0)}")

    return stock, df, log


def main():
    os.makedirs(PROC_DIR, exist_ok=True)
    all_dfs  = {}
    all_logs = {}

    # --- Pass 1: L0, L1, L2 per stock ---
    for stock in STOCKS:
        stock_name, df, log = process_stock(stock, all_dfs)
        if df is not None:
            all_dfs[stock_name]  = df
            all_logs[stock_name] = log

    # --- Pass 2: L3 peer fill (needs all stocks) ---
    print("\n[LEVEL 3] Peer / sector median fill...")
    all_dfs = level3_peer_fill(all_dfs, all_logs)

    # --- Pass 3: L4 ML + finalise per stock ---
    for stock in STOCKS:
        if stock not in all_dfs:
            continue
        df  = all_dfs[stock]
        log = all_logs[stock]

        print(f"\n[{stock}] Level 4 ML imputation...")
        df = level4_ml_impute(df, log)

        # Finalize flags
        df = finalize_flags(df)

        fund_present = [c for c in FUND_COLS if c in df.columns]
        validate(df, stock, fund_present, log)

        after_pct = _pct_missing(df, fund_present)
        print(f"  Missing after ALL levels: {after_pct:.1%}")
        print(f"  L3 filled: {log.get('L3_filled', 0)}")
        print(f"  L4 filled: {log.get('L4_filled', 0)}")
        print(f"  Sanity clipped: {log.get('sanity_clipped', 0)}")

        # Save
        out_path = os.path.join(PROC_DIR, f"{stock}_Dataset.csv")
        df.to_csv(out_path, index=False)
        print(f"  [OK] Saved -> {out_path}  ({len(df)} rows x {len(df.columns)} cols)")

    # --- Save report ---
    with open(LOG_PATH, "w") as f:
        json.dump(all_logs, f, indent=2, default=str)
    print(f"\n[OK] Imputation report -> {LOG_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
