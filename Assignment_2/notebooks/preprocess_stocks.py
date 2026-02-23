"""
preprocess_stocks.py  —  Semantic Fundamental Alignment Pipeline
================================================================
PART A  Build a cross-stock master fundamental schema via
         fuzzy/semantic column matching.
PART B  Construct one aligned, daily-frequency dataset per stock.

Artifacts saved to data/processed/:
  master_fundamental_columns.json  — ordered canonical column list
  column_mapping.json              — {stock: {raw_col: canonical_col}}
  {STOCK}_Dataset.csv              — merged market + fund + macro

Column order in every output CSV:
  Date | market cols | fund_* cols (master schema order) | Macro_* cols

Usage:
    python notebooks/preprocess_stocks.py
"""

import os
import re
import json
import warnings
from difflib import SequenceMatcher
from itertools import combinations
from collections import defaultdict

import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRAPPED_DIR  = os.path.join(BASE_DIR, "data", "scrapped")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Threshold: a metric must appear in this fraction of stocks to enter schema
COVERAGE_THRESHOLD = 0.70

# Fuzzy similarity threshold for column grouping (0–1)
FUZZY_THRESHOLD = 0.82

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# ═════════════════════════════════════════════════════════════════════════════
# PART A — MASTER FUNDAMENTAL SCHEMA
# ═════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# A2 – Column name normalization
# ─────────────────────────────────────────────────────────────────────────────
# Patterns to strip: unit suffixes like (Rs.), %, ₹, (Cr.), etc.
_UNIT_PAT = re.compile(
    r"\(rs\.?\)|\(cr\.?\)|\(inr\)|"
    r"[₹%\(\)\[\]\/\\]|"
    r"\brs\b|\bcr\b|\binr\b|\blakh\b|\bcrore\b|\bper\b|\bshare\b",
    re.IGNORECASE,
)
# Keep only alphanumeric; collapse runs of non-alphanum → single underscore
_CLEAN_PAT = re.compile(r"[^a-z0-9]+")


def normalize_col(name: str) -> str:
    """
    Normalize a raw fundamental column name to a clean snake_case token.

    Steps:
      1. lowercase + strip
      2. remove unit/currency suffixes
      3. replace any run of non-alnum chars with a single underscore
      4. strip leading/trailing underscores

    Example:
        "Net Profit/(Loss) For the Period"  →  "net_profit_loss_for_the_period"
        "Basic EPS (Before Extra Ordinary)" →  "basic_eps_before_extra_ordinary"
        "EPS Before Extra Ordinary - Basic" →  "eps_before_extra_ordinary_basic"
    """
    s = str(name).lower().strip()
    s = _UNIT_PAT.sub(" ", s)
    s = _CLEAN_PAT.sub("_", s)
    s = s.strip("_")
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# A3 – Synonym seeds  (raw normalized → preferred canonical)
# ─────────────────────────────────────────────────────────────────────────────
def build_synonym_map() -> dict[str, str]:
    """
    Return a dictionary mapping known normalized aliases → canonical name.

    The canonical name is the preferred snake_case finance term.
    """
    seeds = {
        # Revenue / top-line
        "net_sales_income_from_operations":     "revenue",
        "net_sales_income_from_operation":      "revenue",
        "total_income_from_operations":         "total_income",
        "total_income_from_operation":          "total_income",
        "interest_earned":                      "revenue",        # bank equivalent

        # Operating costs
        "consumption_of_raw_materials":         "raw_material_cost",
        "purchase_of_traded_goods":             "traded_goods_cost",
        "increase_decrease_in_stocks":          "stock_change",
        "employees_cost":                       "employee_cost",
        "admin_and_selling_expenses":           "admin_selling_expense",

        # Depreciation
        "depreciation":                         "depreciation",

        # P&L line items
        "p_l_before_other_inc_int_excpt_items_tax":  "ebit",
        "p_l_before_other_inc_int_excpt_items_tax_1": "ebit",
        "p_l_before_other_inc":                 "ebit",
        "p_l_before_int_excpt_items_tax":       "ebit_plus_other_income",
        "p_l_before_int":                       "ebit_plus_other_income",
        "other_income":                         "other_income",
        "interest":                             "interest_expense",
        "interest_expended":                    "interest_expense",
        "exceptional_items":                    "exceptional_items",
        "p_l_before_exceptional_items_tax":     "pbt_before_exceptional",
        "p_l_before_tax":                       "pbt",
        "tax":                                  "tax",
        "p_l_after_tax_from_ordinary_activities": "pat_ordinary",
        "net_profit_loss_for_the_period":       "net_profit",
        "minority_interest":                    "minority_interest",
        "share_of_p_l_of_associates":           "associate_share",
        "net_p_l_after_m_i_associates":         "net_profit_after_mi",

        # Balance sheet snippets
        "equity_share_capital":                 "equity_capital",
        "reserves_excluding_revaluation_reserves": "reserves",
        "provisions_and_contingencies":         "provisions",
        "operating_profit_before_provisions_and_contingencies": "operating_profit",

        # EPS variants — all map to basic_eps or diluted_eps
        "basic_eps":                            "basic_eps",
        "basic_eps_before_extra_ordinary":      "basic_eps",
        "eps_before_extra_ordinary_basic":      "basic_eps",
        "diluted_eps":                          "diluted_eps",
        "diluted_eps_before_extra_ordinary":    "diluted_eps",
        "eps_before_extra_ordinary_diluted":    "diluted_eps",
        "basic_eps_after_extra_ordinary":       "basic_eps_after_exceptional",
        "diluted_eps_after_extra_ordinary":     "diluted_eps_after_exceptional",

        # Bank-specific — kept as-is with canonical names
        "a_int_disc_on_adv_bills":              "bank_int_advances",
        "b_income_on_investment":               "bank_int_investments",
        "c_int_on_balances_with_rbi":           "bank_int_rbi",
        "d_others":                             "bank_int_others",
        "gross_npa":                            "gross_npa",
        "net_npa":                              "net_npa",
        "of_gross_npa":                         "gross_npa_pct",
        "of_net_npa":                           "net_npa_pct",
        "other_expenses":                       "other_expenses",
        "exp_capitalised":                      "capitalised_expenses",
    }
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# A3 – Fuzzy grouping of all normalized columns across stocks
# ─────────────────────────────────────────────────────────────────────────────
def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def fuzzy_group_columns(
    all_normalized: list[str],
    synonym_map: dict[str, str],
    threshold: float = FUZZY_THRESHOLD,
) -> dict[str, str]:
    """
    Cluster all_normalized columns into groups by fuzzy similarity,
    then assign each group a canonical name.

    Returns: {normalized_col: canonical_name}
    """
    # 1. Start with synonym overrides
    assignment: dict[str, str] = {}
    for col in all_normalized:
        if col in synonym_map:
            assignment[col] = synonym_map[col]

    # 2. Group remaining by fuzzy similarity
    unassigned = [c for c in all_normalized if c not in assignment]

    # Build Union-Find for transitive grouping
    parent = {c: c for c in unassigned}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for a, b in combinations(unassigned, 2):
        if _similarity(a, b) >= threshold:
            union(a, b)

    # Build groups
    groups: dict[str, list[str]] = defaultdict(list)
    for c in unassigned:
        groups[find(c)].append(c)

    # Pick canonical name: shortest member (usually the base form)
    for members in groups.values():
        canonical = min(members, key=len)
        for m in members:
            assignment[m] = canonical

    return assignment


# ─────────────────────────────────────────────────────────────────────────────
# A4 / A5 / A6 – Compute master schema + per-stock column mapping
# ─────────────────────────────────────────────────────────────────────────────
def build_column_mapping(
    scrapped_dir: str,
    stocks: list[str],
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> tuple[list[str], dict[str, dict[str, str]]]:
    """
    Build the master fundamental schema and per-stock column mapping.

    Returns:
        master_schema  — ordered list of canonical column names
        stock_mapping  — {stock: {raw_col: canonical_col}}
    """
    synonym_map = build_synonym_map()

    # Collect raw indicator names per stock
    raw_per_stock: dict[str, list[str]] = {}
    for stock in stocks:
        path = os.path.join(scrapped_dir, stock, f"{stock}_Fundamental_Data.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        raw_per_stock[stock] = df.iloc[:, 0].dropna().str.strip().tolist()

    # Normalize all names, gather unique set
    norm_per_stock: dict[str, list[str]] = {}
    all_normalized: set[str] = set()
    for stock, raws in raw_per_stock.items():
        normed = [normalize_col(r) for r in raws]
        norm_per_stock[stock] = normed
        all_normalized.update(normed)

    all_normalized_list = sorted(all_normalized)

    # Fuzzy grouping → {norm_col: canonical}
    col_to_canonical = fuzzy_group_columns(all_normalized_list, synonym_map)

    # ── A4: Count canonical coverage across stocks ─────────────────────────
    canonical_stocks: dict[str, set[str]] = defaultdict(set)
    for stock, normed_cols in norm_per_stock.items():
        for nc in normed_cols:
            canon = col_to_canonical.get(nc, nc)
            canonical_stocks[canon].add(stock)

    n_stocks = len(norm_per_stock)
    master_schema = sorted(
        [canon for canon, stk_set in canonical_stocks.items()
         if len(stk_set) / n_stocks >= coverage_threshold]
    )

    # ── A7: Validation ────────────────────────────────────────────────────
    print(f"\n[Schema] {len(all_normalized)} unique normalized columns across {n_stocks} stocks")
    print(f"[Schema] {len(canonical_stocks)} canonical groups found")
    print(f"[Schema] Master schema at >={coverage_threshold:.0%} coverage: {len(master_schema)} columns")

    dropped = [
        (canon, len(stk_set))
        for canon, stk_set in canonical_stocks.items()
        if len(stk_set) / n_stocks < coverage_threshold
    ]
    if dropped:
        print(f"[Schema] Dropped {len(dropped)} low-coverage metrics:")
        for canon, cnt in sorted(dropped, key=lambda x: -x[1]):
            print(f"  - '{canon}' (only {cnt}/{n_stocks} stocks)")

    if len(master_schema) < 5:
        raise RuntimeError(
            f"Master schema has only {len(master_schema)} columns — "
            "something went wrong with fuzzy grouping!"
        )
    if len(master_schema) < 8:
        print(f"[WARN] Master schema has only {len(master_schema)} columns — check synonym seeds.")

    print(f"[Schema] Final schema: {master_schema}")

    # ── A6: Build per-stock raw→canonical mapping ─────────────────────────
    stock_mapping: dict[str, dict[str, str]] = {}
    for stock, raws in raw_per_stock.items():
        mapping: dict[str, str] = {}
        for raw in raws:
            nc = normalize_col(raw)
            canon = col_to_canonical.get(nc, nc)
            if canon in master_schema:
                mapping[raw] = canon
        stock_mapping[stock] = mapping

        unmatched = [r for r in raws if normalize_col(r) not in col_to_canonical or
                     col_to_canonical.get(normalize_col(r)) not in master_schema]
        if unmatched:
            print(f"  [{stock}] {len(unmatched)} cols not in master schema: {unmatched[:5]}")

    return master_schema, stock_mapping


# ═════════════════════════════════════════════════════════════════════════════
# PART B — DATASET CONSTRUCTION PER STOCK
# ═════════════════════════════════════════════════════════════════════════════

# Canonical fund column prefix for output CSV
FUND_PREFIX = "fund_"
# Canonical macro column prefix
MACRO_PREFIX = "Macro_"


def parse_quarter_date(label: str):
    """Convert 'Dec \'25' → pd.Timestamp('2025-12-31')."""
    label = str(label).strip()
    m = re.match(r"([A-Za-z]+)\s*'(\d{2})", label)
    if not m:
        return None
    month = MONTH_MAP.get(m.group(1).lower())
    if month is None:
        return None
    year = 2000 + int(m.group(2))
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


# ─────────────────────────────────────────────────────────────────────────────
# B1 – Loaders
# ─────────────────────────────────────────────────────────────────────────────
def load_fundamental_data(
    stock: str,
    scrapped_dir: str,
    stock_mapping: dict[str, str],
    master_schema: list[str],
) -> pd.DataFrame:
    """
    Load *_Fundamental_Data.csv, transpose wide→long, rename via mapping.

    Applies EFFECTIVE-DATE SHIFT (+1 day) so that:
        Dec '21 quarter end (2021-12-31) → effective 2022-01-01
    This prevents look-ahead bias when merging onto daily market data.

    Returns DataFrame with:
        - 'effective_date'  (shifted date for merge_asof)
        - fund_{canonical}  columns for each metric in master_schema
    """
    path = os.path.join(scrapped_dir, stock, f"{stock}_Fundamental_Data.csv")
    if not os.path.exists(path):
        print(f"  [WARN] No fundamental file for {stock}")
        return pd.DataFrame()

    raw = pd.read_csv(path)
    indicator_col = raw.columns[0]
    quarter_cols  = raw.columns[1:]

    # Set indicator names as index, transpose to (quarter × indicator)
    df = raw.set_index(indicator_col)[quarter_cols].T.copy()
    df.index.name = "quarter_label"
    df.index = df.index.map(parse_quarter_date)
    df = df[df.index.notna()].copy()
    df.index.name = "quarter_end"

    # Rename raw indicators → canonical (only keep master schema hits)
    rename_map = {}
    for raw_col in df.columns:
        canon = stock_mapping.get(str(raw_col).strip())
        if canon and canon in master_schema:
            rename_map[raw_col] = FUND_PREFIX + canon

    df = df.rename(columns=rename_map)
    # Keep only prefixed master columns
    keep = [c for c in df.columns if c.startswith(FUND_PREFIX)]
    df = df[keep]

    # Add missing master columns as NaN
    for canon in master_schema:
        col = FUND_PREFIX + canon
        if col not in df.columns:
            df[col] = float("nan")

    # Convert to numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_index()

    # ── B3: Effective-date shift (+1 day = first day of next quarter)
    # E.g. Dec 31 2021 → Jan 1 2022  (values valid from Jan 1 onwards)
    df["effective_date"] = df.index + pd.Timedelta(days=1)
    df = df.reset_index(drop=True)

    return df


def load_market_data(stock: str, scrapped_dir: str) -> pd.DataFrame:
    """Load *_Market_Data.csv — parse Date, sort, dedup."""
    path = os.path.join(scrapped_dir, stock, f"{stock}_Market_Data.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Market data missing: {path}")
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return df


def load_macro_data(scrapped_dir: str) -> pd.DataFrame:
    """Load macro_indicators.csv — parse date, sort, dedup, prefix Macro_."""
    path = os.path.join(scrapped_dir, "macro_indicators.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Macro file missing: {path}")
    df = pd.read_csv(path)
    # Drop unnamed index col
    unnamed = [c for c in df.columns if not c.strip() or c.startswith("Unnamed")]
    df = df.drop(columns=unnamed)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    rename = {c: MACRO_PREFIX + c for c in df.columns if c != "date"}
    df = df.rename(columns=rename)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# B2–B6 – Full merge pipeline for one stock
# ─────────────────────────────────────────────────────────────────────────────
def merge_stock_dataset(
    stock: str,
    scrapped_dir: str,
    processed_dir: str,
    stock_mapping: dict[str, str],
    master_schema: list[str],
    macro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the aligned daily dataset for one stock.

    Merges:
      market (daily base)
        ← fundamentals via merge_asof(effective_date, direction=backward)
        ← macro    via merge_asof(date, direction=backward)

    Then:
      - forward-fills fund_* and Macro_* columns
      - enforces B6 column order: Date | market | fund_* | Macro_*
    """
    print(f"\n[{stock}] Loading data...")

    market = load_market_data(stock, scrapped_dir)
    print(f"  Market : {len(market)} rows  [{market['Date'].min().date()} -> {market['Date'].max().date()}]")

    fund = load_fundamental_data(stock, scrapped_dir, stock_mapping, master_schema)
    if fund.empty:
        # Create empty fund frame with all master schema columns
        fund = pd.DataFrame(columns=["effective_date"] + [FUND_PREFIX + c for c in master_schema])
        print(f"  Fund   : (none)")
    else:
        print(f"  Fund   : {len(fund)} quarters | "
              f"effective {fund['effective_date'].min().date()} -> {fund['effective_date'].max().date()}")

    print(f"  Macro  : {len(macro)} rows")

    # ── B4a: Merge fundamentals onto market (as-of, backward) ─────────────
    fund_sorted   = fund.sort_values("effective_date")
    market_sorted = market.sort_values("Date")

    if not fund.empty:
        merged = pd.merge_asof(
            market_sorted,
            fund_sorted,
            left_on="Date",
            right_on="effective_date",
            direction="backward",
        )
        merged = merged.drop(columns=["effective_date"], errors="ignore")
    else:
        merged = market_sorted.copy()
        for canon in master_schema:
            merged[FUND_PREFIX + canon] = float("nan")

    # ── B4b: Merge macro onto market (as-of, backward) ────────────────────
    macro_sorted = macro.sort_values("date")
    merged = pd.merge_asof(
        merged.sort_values("Date"),
        macro_sorted,
        left_on="Date",
        right_on="date",
        direction="backward",
    )
    merged = merged.drop(columns=["date"], errors="ignore")

    # ── B3/B5: Forward-fill quarterly fundamentals + macro gaps ──────────
    fund_cols  = [c for c in merged.columns if c.startswith(FUND_PREFIX)]
    macro_cols = [c for c in merged.columns if c.startswith(MACRO_PREFIX)]
    merged = merged.sort_values("Date").reset_index(drop=True)
    merged[fund_cols + macro_cols] = merged[fund_cols + macro_cols].ffill()

    # ── B5: Enforce numeric dtypes on fundamental columns ─────────────────
    for col in fund_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # ── B6: Strict column ordering ─────────────────────────────────────────
    # Date | market cols | fund_* (master schema order) | Macro_* (sorted)
    market_cols = [c for c in merged.columns
                   if c != "Date" and not c.startswith(FUND_PREFIX) and not c.startswith(MACRO_PREFIX)]
    ordered_fund  = [FUND_PREFIX + c for c in master_schema]
    ordered_macro = sorted(macro_cols)
    final_cols = ["Date"] + market_cols + ordered_fund + ordered_macro
    merged = merged[[c for c in final_cols if c in merged.columns]]

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, f"{stock}_Dataset.csv")
    merged.to_csv(out_path, index=False)
    print(f"  [OK] Saved -> {out_path}  ({merged.shape[0]} rows x {merged.shape[1]} cols)")

    return merged


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def discover_stocks(scrapped_dir: str) -> list[str]:
    """Return sorted list of stock tickers that have a Market_Data file."""
    stocks = []
    for entry in os.scandir(scrapped_dir):
        if entry.is_dir():
            mf = os.path.join(entry.path, f"{entry.name}_Market_Data.csv")
            if os.path.exists(mf):
                stocks.append(entry.name)
    return sorted(stocks)


def main():
    print("=" * 65)
    print("Stock Dataset Consolidation — Semantic Alignment Pipeline")
    print("=" * 65)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    stocks = discover_stocks(SCRAPPED_DIR)
    print(f"Found {len(stocks)} stocks: {', '.join(stocks)}")

    # ── PART A: Build master schema ───────────────────────────────────────
    print("\n[PART A] Building master fundamental schema...")
    master_schema, stock_mapping = build_column_mapping(
        SCRAPPED_DIR, stocks, COVERAGE_THRESHOLD
    )

    # A8: Persist artifacts
    schema_path  = os.path.join(PROCESSED_DIR, "master_fundamental_columns.json")
    mapping_path = os.path.join(PROCESSED_DIR, "column_mapping.json")

    with open(schema_path, "w") as f:
        json.dump(master_schema, f, indent=2)
    with open(mapping_path, "w") as f:
        json.dump(stock_mapping, f, indent=2)

    print(f"\n[PART A] Artifacts saved:")
    print(f"  {schema_path}")
    print(f"  {mapping_path}")

    # ── PART B: Construct per-stock datasets ─────────────────────────────
    print("\n[PART B] Building per-stock datasets...")
    # Load macro once (shared across all stocks)
    macro = load_macro_data(SCRAPPED_DIR)

    results, errors = {}, []
    for stock in stocks:
        s_mapping = stock_mapping.get(stock, {})
        try:
            df = merge_stock_dataset(
                stock, SCRAPPED_DIR, PROCESSED_DIR,
                s_mapping, master_schema, macro,
            )
            results[stock] = df.shape
        except Exception as exc:
            print(f"  [ERR] {stock}: {exc}")
            errors.append((stock, str(exc)))

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("Summary")
    print("=" * 65)
    print(f"Master schema ({len(master_schema)} cols): {master_schema}")
    print()
    for stock, shape in results.items():
        print(f"  {stock:<20}  {shape[0]:>6} rows x {shape[1]:>3} cols")
    if errors:
        print("\nErrors:")
        for stock, msg in errors:
            print(f"  {stock}: {msg}")

    # ── Cross-stock consistency check ─────────────────────────────────────
    if len(results) > 1:
        fund_sets = {}
        for stock in results:
            path = os.path.join(PROCESSED_DIR, f"{stock}_Dataset.csv")
            cols = pd.read_csv(path, nrows=0).columns.tolist()
            fund_sets[stock] = frozenset(c for c in cols if c.startswith(FUND_PREFIX))
        unique_sets = set(fund_sets.values())
        if len(unique_sets) == 1:
            print("\n[OK] All stocks have IDENTICAL fundamental schema.")
        else:
            print("\n[WARN] Schema mismatch detected:")
            ref = next(iter(fund_sets.values()))
            for stock, fset in fund_sets.items():
                diff = fset.symmetric_difference(ref)
                if diff:
                    print(f"  {stock}: extra/missing = {diff}")

    print("\nDone.")


if __name__ == "__main__":
    main()
