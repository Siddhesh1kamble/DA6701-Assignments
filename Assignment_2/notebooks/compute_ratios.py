"""
compute_ratios.py  —  Financial Ratio Computation Pipeline
===========================================================
PART A  Robust column resolution:
          exact match → synonym lookup → difflib fuzzy (≥85%) → NaN placeholder

PART B  11 vectorized financial ratios appended to each processed dataset.

Inputs:   data/processed/{STOCK}_Dataset.csv
Outputs:  data/processed/{STOCK}_Dataset.csv      (overwritten with ratio cols)
          data/processed/column_resolution_log.json
          data/processed/ratio_validation_log.json

Usage:
    python notebooks/compute_ratios.py
"""

import os
import re
import json
import glob
import warnings
from difflib import SequenceMatcher, get_close_matches
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# ─────────────────────────────────────────────────────────────────────────────
# Exact names expected by each ratio (the "canonical ratio inputs")
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED_INPUTS = [
    "close_price",
    "eps",
    "total_debt",
    "equity",
    "net_income",
    "assets",
    "revenue",
    "ebit",
    "shares_outstanding",
    "current_assets",
    "current_liabilities",
    "inventory",
    "fcf",
]

# ─────────────────────────────────────────────────────────────────────────────
# Final ratio column names (strict — must match across all stocks)
# ─────────────────────────────────────────────────────────────────────────────
RATIO_COLUMNS = [
    "P/E Ratio",
    "Debt-to-Equity",
    "ROE",
    "Return on Assets",
    "Net Profit Margin",
    "Operating Margin",
    "Price-to-Book",
    "Current Ratio",
    "Asset Turnover",
    "Inventory Turnover",
    "FCF Yield",
]

# Fuzzy acceptance threshold (0–1)
FUZZY_THRESHOLD = 0.85


# ═════════════════════════════════════════════════════════════════════════════
# PART A — COLUMN RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# A2 — Normalization (shared with preprocess_stocks, inline here for autonomy)
# ─────────────────────────────────────────────────────────────────────────────
_CLEAN = re.compile(r"[^a-z0-9]+")


def normalize_col(name: str) -> str:
    """
    Normalize a column name token for matching.

    'Fund_Net Profit/(Loss)'  →  'fund_net_profit_loss'
    'close'                   →  'close'
    """
    s = str(name).lower().strip()
    s = _CLEAN.sub("_", s)
    s = s.strip("_")
    s = re.sub(r"_+", "_", s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# A3 — Synonym dictionary
#      Keys   = canonical ratio input name (as in REQUIRED_INPUTS)
#      Values = list of column-name substrings / normalized aliases to check
# ─────────────────────────────────────────────────────────────────────────────
def build_synonym_dict() -> dict[str, list[str]]:
    """
    Map each canonical ratio input to a priority-ordered list of known aliases.

    The resolver checks these against *normalized* dataset column names.
    Order matters: first hit wins.
    """
    return {
        "close_price": [
            "close",
            "closing_price",
            "adj_close",
            "adjusted_close",
            "ltp",
        ],
        "eps": [
            "fund_basic_eps",
            "fund_diluted_eps",
            "basic_eps",
            "diluted_eps",
            "earnings_per_share",
            "eps_before_extra_ordinary",
        ],
        "net_income": [
            "fund_net_profit",
            "fund_pat_ordinary",
            "fund_net_profit_after_mi",
            "net_profit",
            "pat",
            "profit_after_tax",
            "net_income",
        ],
        "revenue": [
            "fund_revenue",
            "fund_total_income",
            "net_sales",
            "total_revenue",
            "turnover",
            "sales",
        ],
        "ebit": [
            "fund_ebit",
            "fund_ebit_plus_other_income",
            "operating_profit",
            "ebit",
        ],
        "equity": [
            "fund_equity_capital",
            "shareholders_equity",
            "net_worth",
            "book_value",
            "equity_capital",
        ],
        "total_debt": [
            "total_debt",
            "borrowings",
            "long_term_debt",
            "debt",
            "total_borrowings",
        ],
        "assets": [
            "total_assets",
            "assets",
            "total_asset",
            "balance_sheet_total",
        ],
        "shares_outstanding": [
            "shares_outstanding",
            "no_of_shares",
            "weighted_avg_shares",
            "shares",
        ],
        "current_assets": [
            "current_assets",
            "current_asset",
        ],
        "current_liabilities": [
            "current_liabilities",
            "current_liability",
        ],
        "inventory": [
            "inventory",
            "inventories",
            "stock_in_trade",
        ],
        "fcf": [
            "fcf",
            "free_cash_flow",
            "cash_flow_from_operations",
            "operating_cash_flow",
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# A4 / A5 — Fuzzy fallback
# ─────────────────────────────────────────────────────────────────────────────
def _fuzzy_best(query: str, candidates: list[str], threshold: float) -> tuple[str | None, float]:
    """
    Return (best_match, score) for query against candidates.
    Uses SequenceMatcher for deterministic, library-free matching.
    Returns (None, 0.0) if no candidate meets threshold.
    """
    best_col, best_score = None, 0.0
    for cand in candidates:
        score = SequenceMatcher(None, query, cand).ratio()
        if score > best_score:
            best_score, best_col = score, cand
    if best_score >= threshold:
        return best_col, best_score
    return None, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Main resolver — A7: Resolution Logging built in
# ─────────────────────────────────────────────────────────────────────────────
def resolve_columns(
    df: pd.DataFrame,
    required_inputs: list[str],
    synonym_dict: dict[str, list[str]],
    fuzzy_threshold: float = FUZZY_THRESHOLD,
) -> tuple[dict[str, str | None], dict[str, dict]]:
    """
    Resolve each canonical ratio input to a column name in df.

    Resolution order (A3 → A4 priority):
      1. Exact match on normalized column names
      2. Synonym lookup (normalized alias in dataset columns)
      3. Fuzzy match on normalized names (≥ threshold)
      4. Unresolved → None (column will be filled with NaN)

    A8: Column normalization runs once up-front; fuzzy never runs per-row.

    Returns:
        resolved   — {canonical_input: actual_column_name_or_None}
        log_detail — {canonical_input: {method, matched_col, score}}
    """
    # Build normalized → original mapping for dataset columns (once per call)
    norm_to_orig: dict[str, str] = {}
    for col in df.columns:
        nc = normalize_col(col)
        norm_to_orig[nc] = col  # last one wins if collision (rare)

    norm_cols_set = set(norm_to_orig.keys())

    resolved: dict[str, str | None] = {}
    log_detail: dict[str, dict] = {}

    for canonical in required_inputs:
        nc_canonical = normalize_col(canonical)

        # ── Step 1: Exact match ───────────────────────────────────────────
        if nc_canonical in norm_cols_set:
            actual = norm_to_orig[nc_canonical]
            resolved[canonical] = actual
            log_detail[canonical] = {"method": "exact", "matched": actual, "score": 1.0}
            continue

        # ── Step 2: Synonym lookup ────────────────────────────────────────
        hit = None
        for alias in synonym_dict.get(canonical, []):
            nc_alias = normalize_col(alias)
            if nc_alias in norm_cols_set:
                hit = norm_to_orig[nc_alias]
                log_detail[canonical] = {"method": "synonym", "matched": hit,
                                         "alias_used": alias, "score": 1.0}
                break
        if hit:
            resolved[canonical] = hit
            continue

        # ── Step 3: Fuzzy fallback ────────────────────────────────────────
        best_col, best_score = _fuzzy_best(nc_canonical, list(norm_cols_set), fuzzy_threshold)
        if best_col:
            actual = norm_to_orig[best_col]
            # A5: log warning if multiple candidates were close
            n_close = sum(
                1 for c in norm_cols_set
                if SequenceMatcher(None, nc_canonical, c).ratio() >= fuzzy_threshold
            )
            if n_close > 1:
                print(f"  [WARN] Ambiguous fuzzy match for '{canonical}': "
                      f"{n_close} candidates, chose '{actual}' (score={best_score:.2f})")
            resolved[canonical] = actual
            log_detail[canonical] = {"method": "fuzzy", "matched": actual, "score": round(best_score, 3)}
            continue

        # ── Step 4: Unresolved ────────────────────────────────────────────
        resolved[canonical] = None
        log_detail[canonical] = {"method": "unresolved", "matched": None,
                                  "best_candidate_score": round(best_score, 3)}
        print(f"  [INFO] '{canonical}' not found — will use NaN (best fuzzy={best_score:.2f})")

    return resolved, log_detail


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get column from df or NaN series
# ─────────────────────────────────────────────────────────────────────────────
def _get(df: pd.DataFrame, col: str | None) -> pd.Series:
    """Return the named series as float64, or a NaN series if col is None."""
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[col].astype("float64")


def _safe_div(num: pd.Series, denom: pd.Series, guard_zero: bool = True) -> pd.Series:
    """
    Vectorized safe division.

    - Replaces ±inf with NaN
    - If guard_zero: masks rows where denom ≤ 0 with NaN
    """
    if guard_zero:
        denom = denom.where(denom > 0, other=np.nan)
    result = num / denom
    return result.replace([np.inf, -np.inf], np.nan).astype("float64")


# ═════════════════════════════════════════════════════════════════════════════
# PART B — RATIO COMPUTATIONS
# ═════════════════════════════════════════════════════════════════════════════

def compute_ratios(
    df: pd.DataFrame,
    resolved: dict[str, str | None],
) -> pd.DataFrame:
    """
    Append all 11 ratio columns to df in-place, using resolved column mapping.

    All operations are fully vectorized — no row iteration.
    Numerically safe: div-by-zero → NaN, inf → NaN, dtype = float64.
    """
    # Pull resolved series (NaN series if missing)
    close       = _get(df, resolved["close_price"])
    eps         = _get(df, resolved["eps"])
    total_debt  = _get(df, resolved["total_debt"])
    equity      = _get(df, resolved["equity"])
    net_income  = _get(df, resolved["net_income"])
    assets      = _get(df, resolved["assets"])
    revenue     = _get(df, resolved["revenue"])
    ebit        = _get(df, resolved["ebit"])
    shares      = _get(df, resolved["shares_outstanding"])
    cur_assets  = _get(df, resolved["current_assets"])
    cur_liab    = _get(df, resolved["current_liabilities"])
    inventory   = _get(df, resolved["inventory"])
    fcf         = _get(df, resolved["fcf"])

    # ── 1. P/E Ratio  (annualise quarterly EPS × 4, guard EPS ≤ 0) ─────────
    annual_eps = (eps * 4).where(eps > 0, other=np.nan)
    df["P/E Ratio"] = _safe_div(close, annual_eps)

    # ── 2. Debt-to-Equity  (guard equity ≤ 0) ───────────────────────────────
    df["Debt-to-Equity"] = _safe_div(total_debt, equity)

    # ── 3. ROE  (annualised net_income, guard equity ≤ 0) ───────────────────
    df["ROE"] = _safe_div(net_income * 4, equity)

    # ── 4. Return on Assets  (annualised, guard assets ≤ 0) ─────────────────
    df["Return on Assets"] = _safe_div(net_income * 4, assets)

    # ── 5. Net Profit Margin  (no annualisation; guard revenue ≤ 0) ─────────
    df["Net Profit Margin"] = _safe_div(net_income, revenue)

    # ── 6. Operating Margin  (guard revenue ≤ 0) ────────────────────────────
    df["Operating Margin"] = _safe_div(ebit, revenue)

    # ── 7. Price-to-Book ─────────────────────────────────────────────────────
    #   book_value_per_share = equity / shares;  P/B = close / bvps
    bvps = _safe_div(equity, shares)
    df["Price-to-Book"] = _safe_div(close, bvps)

    # ── 8. Current Ratio  (guard current_liabilities ≤ 0) ──────────────────
    df["Current Ratio"] = _safe_div(cur_assets, cur_liab)

    # ── 9. Asset Turnover  (annualised revenue, guard assets ≤ 0) ───────────
    df["Asset Turnover"] = _safe_div(revenue * 4, assets)

    # ── 10. Inventory Turnover  (annualised revenue, guard inventory ≤ 0) ───
    df["Inventory Turnover"] = _safe_div(revenue * 4, inventory)

    # ── 11. FCF Yield  (annualised FCF / market_cap; guard mktcap ≤ 0) ─────
    market_cap = (close * shares).where((close * shares) > 0, other=np.nan)
    df["FCF Yield"] = _safe_div(fcf * 4, market_cap, guard_zero=False)

    # ── Enforce float64 on all ratio cols ────────────────────────────────────
    for col in RATIO_COLUMNS:
        df[col] = df[col].astype("float64")

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Validation helper
# ═════════════════════════════════════════════════════════════════════════════

def validate_ratios(df: pd.DataFrame, stock: str) -> dict:
    """
    Assert correctness of ratio columns; return a validation record.

    Checks:
      - All RATIO_COLUMNS present
      - No ±inf values
      - Row count unchanged (passed in as len(df))
    """
    missing_cols = [c for c in RATIO_COLUMNS if c not in df.columns]
    if missing_cols:
        raise AssertionError(f"[{stock}] Missing ratio columns: {missing_cols}")

    inf_counts = {}
    for col in RATIO_COLUMNS:
        n_inf = np.isinf(df[col].fillna(0)).sum()
        if n_inf > 0:
            print(f"  [WARN] {col} has {n_inf} inf values — replacing with NaN")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        inf_counts[col] = int(n_inf)

    nan_pcts = {
        col: round(float(df[col].isna().mean() * 100), 1)
        for col in RATIO_COLUMNS
    }

    record = {
        "rows":      len(df),
        "nan_pct":   nan_pcts,
        "inf_found": inf_counts,
    }
    return record


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("Financial Ratio Computation Pipeline")
    print("=" * 65)

    # Discover processed stock files
    pattern = os.path.join(PROCESSED_DIR, "*_Dataset.csv")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"ERROR: No _Dataset.csv files found in {PROCESSED_DIR}")
        return

    stocks = [os.path.basename(f).replace("_Dataset.csv", "") for f in files]
    print(f"Found {len(stocks)} datasets: {', '.join(stocks)}\n")

    synonym_dict = build_synonym_dict()

    resolution_log:  dict[str, dict] = {}
    validation_log:  dict[str, dict] = {}

    for filepath, stock in zip(files, stocks):
        print(f"[{stock}]")

        # ── Load ─────────────────────────────────────────────────────────
        df = pd.read_csv(filepath, parse_dates=["Date"])
        original_rows = len(df)

        # ── A: Resolve columns (once per dataset — A8 perf constraint) ───
        resolved, log_detail = resolve_columns(
            df, REQUIRED_INPUTS, synonym_dict, FUZZY_THRESHOLD
        )
        resolution_log[stock] = log_detail

        # A7: summarise resolution
        methods = defaultdict(list)
        for inp, detail in log_detail.items():
            methods[detail["method"]].append(inp)
        print(f"  Exact={len(methods['exact'])} | "
              f"Synonym={len(methods['synonym'])} | "
              f"Fuzzy={len(methods['fuzzy'])} | "
              f"Unresolved={len(methods['unresolved'])}")
        if methods["unresolved"]:
            print(f"  Unresolved: {methods['unresolved']}")

        # ── B: Compute ratios ─────────────────────────────────────────────
        df = compute_ratios(df, resolved)

        # ── Validate ──────────────────────────────────────────────────────
        val = validate_ratios(df, stock)
        assert len(df) == original_rows, f"Row count changed! {original_rows} -> {len(df)}"
        validation_log[stock] = val

        # Print NaN% for each ratio
        print(f"  Ratio NaN%: " + " | ".join(
            f"{col}={pct}%" for col, pct in val["nan_pct"].items()
        ))

        # ── Save (overwrite) ──────────────────────────────────────────────
        df.to_csv(filepath, index=False)
        print(f"  [OK] Saved -> {filepath}  ({df.shape[0]} rows x {df.shape[1]} cols)\n")

    # ── Save JSON artifacts ───────────────────────────────────────────────
    res_log_path = os.path.join(PROCESSED_DIR, "column_resolution_log.json")
    val_log_path = os.path.join(PROCESSED_DIR, "ratio_validation_log.json")

    with open(res_log_path, "w") as f:
        json.dump(resolution_log, f, indent=2)
    with open(val_log_path, "w") as f:
        json.dump(validation_log, f, indent=2)

    print("=" * 65)
    print("Artifacts saved:")
    print(f"  {res_log_path}")
    print(f"  {val_log_path}")

    # ── Final cross-stock consistency check ───────────────────────────────
    ratio_sets = set()
    for filepath in files:
        cols = pd.read_csv(filepath, nrows=0).columns.tolist()
        ratio_sets.add(frozenset(c for c in cols if c in RATIO_COLUMNS))

    if len(ratio_sets) == 1:
        print("\n[OK] All stocks have IDENTICAL ratio schema.")
    else:
        print("\n[WARN] Ratio schema mismatch across stocks!")

    print("\nDone.")


if __name__ == "__main__":
    main()
