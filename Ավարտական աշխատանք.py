#1
"""
ԿՐԻՊՏՈԱԿՏԻՎՆԵՐԻ ՇՈՒԿԱ — ԱՄԲՈՂՋԱԿԱՆ ԶՈՒՅԳԱՅԻՆ ԿՈՐԵԼԱՑԻՈՆ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ
Լավագույն 300 ակտիվները | 2018-01-01 - 2026-01-01 
Տվյալների աղբյուրներ. CoinGecko (Վարկանիշներ) + Yahoo Finance (Պատմական գներ)
 
  pip install requests pandas numpy matplotlib seaborn tqdm pyarrow scipy yfinance
"""

import sys, time, logging, warnings
import requests
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path
from datetime import datetime, timezone
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
matplotlib.rcParams["figure.max_open_warning"] = 0

# =============================================================================
#  ԿԱՐԳԱՎՈՐՈՒՄՆԵՐ
# =============================================================================

START_DATE            = "2018-01-01"
END_DATE              = "2026-01-01"

TOP_N                 = 300
MIN_DATA_COVERAGE     = 0.90      # նվազագույնը 90% ծածկույթ
STABLECOIN_STD_THRESH = 0.010     # օրական լոգ-եկամտաբերության ստանդարտ շեղում < 1% => սթեյբլքոին
MIN_MARKET_CAP_USD    = 1_000_000

TOP_K = 5   # յուրաքանչյուր ակտիվի համար՝ TOP ամենաշատ + ամենաքիչ կորելացված ակտիվները

COINGECKO_BASE   = "https://api.coingecko.com/api/v3"
CG_API_KEY       = ""
RATE_LIMIT_DELAY = 2.5
MAX_RETRIES      = 8
BACKOFF_FACTOR   = 2.0

CACHE_DIR     = Path("crypto_cache");   CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR   = Path("crypto_results"); RESULTS_DIR.mkdir(exist_ok=True)
PER_ASSET_DIR = RESULTS_DIR / "per_asset_top_correlations"
PER_ASSET_DIR.mkdir(exist_ok=True)

# =============================================================================
#  ՀԱՅՏՆԻ ՍԹԵՅԲԼՔՈԻՆՆԵՐ
# =============================================================================

KNOWN_STABLECOINS = {
    "usdt","usdc","busd","dai","tusd","usdp","usdd","frax","lusd","susd",
    "fei","mim","alusd","cusd","ust","ustc","usdb","usdy","gusd","husd",
    "usdk","usdx","xusd","usdm","usde","usds","rlusd","fdusd","pyusd",
    "crvusd","gho","dola","musd","ousd","mkusd","usd0","usdz","rusd",
    "nusd","zusd","usdl","dusd","volt","vst",
    "eurs","eurt","ageur","seur","steur","jeur","eurc","eure","gbpt",
    "cadc","xsgd","bidr","brl","idrt","xchf","tryb","brz","gyen",
    "nzds","xaud","mxnt","jpyc","thbx","krw",
    "paxg","xaut","dgld","pmgt","slvt","cache","ounz",
    "wbtc","renbtc","sbtc","hbtc","tbtc","cbbtc","lbtc","btcb",
    "steth","wsteth","reth","cbeth","ankraeth","sfrxeth",
    "oseth","lseth","meth","sweth","weeth","rseth","ezeth","pufeth",
    "weth","wbnb","wmatic","wavax","wsol","wftm","wone","wcro",
    "bean","float","usdv","usn","rai",
}

# =============================================================================
#  LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / "analysis.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# (CoinGecko-ի լավագույնների ցանկ)
# =============================================================================

def _build_session():
    s = requests.Session()
    retry = Retry(
        total=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    if CG_API_KEY:
        s.headers["x-cg-demo-api-key"] = CG_API_KEY
    s.headers.update({"Accept": "application/json",
                      "User-Agent": "CryptoResearch/3.1"})
    return s

SESSION = _build_session()


def cg_get(endpoint, params=None, _r=0):
    url = f"{COINGECKO_BASE}/{endpoint}"
    try:
        resp = SESSION.get(url, params=params, timeout=45)
        time.sleep(RATE_LIMIT_DELAY)
        if resp.status_code == 429:
            wait = RATE_LIMIT_DELAY * (3 ** (_r + 1))
            log.warning(f"Հարցումների սահմանափակում (Rate-limited) => սպասում {wait:.1f} վրկ")
            time.sleep(wait)
            return cg_get(endpoint, params, _r + 1) if _r < MAX_RETRIES else None
        if resp.status_code == 200:
            return resp.json()
        log.warning(f"HTTP {resp.status_code} — {endpoint}")
        if _r < 3:
            time.sleep(RATE_LIMIT_DELAY * 2)
            return cg_get(endpoint, params, _r + 1)
        return None
    except requests.RequestException as e:
        log.error(f"Հարցման սխալ (RequestException) ({endpoint}): {e}")
        if _r < MAX_RETRIES:
            time.sleep(RATE_LIMIT_DELAY * 2)
            return cg_get(endpoint, params, _r + 1)
        return None
    
# =============================================================================
#  ՏՎՅԱԼՆԵՐԻ ԲԵՌՆՈՒՄ
# =============================================================================

def get_top_n_coins(n=TOP_N):
    cf = CACHE_DIR / f"top_{n}_coins.parquet"
    if cf.exists():
        log.info(f"Տվյալների վերականգնում քեշից (մետաղադրամների ցանկ). {cf}")
        return pd.read_parquet(cf)

    log.info(f"Բեռնվում են լավագույն {n} մետաղադրամները CoinGecko-ից ...")
    coins = []
    per_page = 250
    for page in range(1, (n // per_page) + 2):
        data = cg_get("coins/markets", {
            "vs_currency": "usd", "order": "market_cap_desc",
            "per_page": per_page, "page": page, "sparkline": "false",
        })
        if not data:
            log.error(f"Էջ {page}-ի բեռնումը ձախողվեց"); break
        coins.extend(data)
        log.info(f"  Էջ {page}: +{len(data)} (ընդհանուր {len(coins)})")
        if len(coins) >= n:
            break

    if not coins:
        raise RuntimeError("Հնարավոր չէ բեռնել մետաղադրամների ցանկը։")

    df = pd.DataFrame(coins).head(n)
    cols = [c for c in ["id", "symbol", "name", "market_cap", "total_volume"] if c in df.columns]
    df = df[cols].copy()
    df["symbol"] = df["symbol"].str.lower().str.strip()
    df = df.dropna(subset=["id", "symbol"]).reset_index(drop=True)
    df.to_parquet(cf, index=False)
    log.info(f"Մետաղադրամների ցանկը պահպանվեց ({len(df)}) => {cf}")
    return df

# =============================================================================
#  ՖԻԼՏՐԱՑՄԱՆ ԳՈՐԾԸՆԹԱՑ
# =============================================================================

def f1_remove_known_stablecoins(df):
    before = len(df)
    df = df[~df["symbol"].isin(KNOWN_STABLECOINS)].copy().reset_index(drop=True)
    log.info(f"[F-1] Հայտնի սթեյբլքոինները հեռացված են : {before - len(df):>3d}  =>  մնաց {len(df)}")
    return df

def f2_remove_low_mcap(df):
    before = len(df)
    df = df[df["market_cap"].fillna(0) >= MIN_MARKET_CAP_USD].copy().reset_index(drop=True)
    log.info(f"[F-2] Ցածր շուկայական կապիտալիզացիայով ակտիվները հեռացված են : {before - len(df):>3d}  =>  մնաց {len(df)}")
    return df

def f3_build_price_matrix(df):
    ticker_to_id = {}
    valid_tickers = []
    for _, row in df.iterrows():
        sym = str(row["symbol"]).upper()
        yf_ticker = f"{sym}-USD"
        if yf_ticker not in ticker_to_id:
            ticker_to_id[yf_ticker] = row["id"]
            valid_tickers.append(yf_ticker)

    log.info(f"Բեռնվում են պատմական գները {len(valid_tickers)} ակտիվների համար Yahoo Finance-ի միջոցով...")
    try:
        data = yf.download(valid_tickers, start=START_DATE, end=END_DATE, interval="1d", progress=False)
        if data.empty:
            log.error("Yahoo Finance-ը վերադարձրել է դատարկ տվյալներ։")
            return pd.DataFrame()

        if isinstance(data.columns, pd.MultiIndex):
            if 'Close' in data.columns.levels[0]:
                prices = data['Close']
            else:
                prices = data
        else:
            prices = data

        prices = prices.rename(columns=ticker_to_id)
        keep_cols = [c for c in prices.columns if c in df["id"].values]
        prices = prices[keep_cols]
        prices = prices.dropna(axis=1, how='all')
        prices.index = pd.to_datetime(prices.index, utc=True).normalize()
        log.info(f"Գների սկզբնական մատրից. {prices.shape[0]} օր x {prices.shape[1]} մետաղադրամ")
        return prices

    except Exception as e:
        log.error(f"Չհաջողվեց բեռնել տվյալներ Yahoo Finance-ից. {e}")
        return pd.DataFrame()

def f4_filter_coverage(mat):
    full_range = pd.date_range(
        pd.Timestamp(START_DATE, tz="UTC"),
        pd.Timestamp(END_DATE,   tz="UTC") - pd.Timedelta(days=1),
        freq="D",
    )
    required = int(len(full_range) * MIN_DATA_COVERAGE)
    counts   = mat.reindex(full_range).notna().sum()
    keep     = counts[counts >= required].index
    dropped  = mat.shape[1] - len(keep)
    log.info(f"[F-4] <{MIN_DATA_COVERAGE*100:.0f}% ծածկույթ ունեցողները հեռացված են : {dropped:>3d}  =>  մնաց {len(keep)}")
    return mat[keep]

def f5_filter_stat_stablecoins(mat):
    lr  = np.log(mat / mat.shift(1))
    std = lr.std()
    rem = std[std < STABLECOIN_STD_THRESH].index.tolist()
    rem += std[std.isna()].index.tolist()
    rem = list(set(rem))
    log.info(f"[F-5] Վիճակագրական սթեյբլքոիններ/մեռած մետաղադրամներ հեռացված են : {len(rem):>3d}  =>  մնաց {mat.shape[1] - len(rem)}")
    return mat.drop(columns=rem, errors="ignore")

# =============================================================================
#  ԿՈՐԵԼԱՑԻԱ
# =============================================================================

def compute_log_returns(mat):
    filled = mat.ffill(limit=3)
    lr     = np.log(filled / filled.shift(1))
    lr     = lr.dropna(thresh=int(lr.shape[1] * 0.5))
    return lr

def compute_correlation_matrix(lr):
    log.info(f"Հաշվարկվում է Պիրսոնի կորելացիոն մատրիցը {lr.shape[1]}x{lr.shape[1]} ...")
    corr = lr.corr(method="pearson", min_periods=60)
    log.info("Կորելացիոն մատրիցը պատրաստ է։")
    return corr


# =============================================================================
#  ԱՄԵՆԱԲԱՐՁՐ K-Ն ՅՈՒՐԱՔԱՆՉՅՈՒՐ ԱԿՏԻՎԻ ՀԱՄԱՐ
# =============================================================================

def compute_all_top_k(corr: pd.DataFrame, meta: pd.DataFrame, k: int = TOP_K) -> pd.DataFrame:
    id2sym  = {str(row["id"]): str(row["symbol"]).upper() for _, row in meta.iterrows()}
    id2name = {str(row["id"]): str(row["name"]) for _, row in meta.iterrows()}
    all_assets = [str(c) for c in corr.columns]

    if len(all_assets) == 0:
        log.error("Կորելացիոն մատրիցն ունի 0 սյունակ — մշակելու ոչինչ չկա։")
        return pd.DataFrame()

    log.info(f"Հաշվարկվում են ամենաբարձր {k} կորելացիաները {len(all_assets)} ակտիվների համար ...")
    rows = []
    for asset in tqdm(all_assets, desc="Ըստ ակտիվի լավագույն K", unit="ակտիվ"):
        col = corr[asset].copy()
        col.index = [str(c) for c in col.index]
        if asset in col.index: col = col.drop(index=asset)
        col = col.dropna()
        if col.empty: continue

        sym  = id2sym.get(asset,  asset.upper())
        name = id2name.get(asset, asset)

        for rank, (peer, val) in enumerate(col.nlargest(k).items(), 1):
            rows.append({
                "asset_id": asset, "asset_symbol": sym, "asset_name": name,
                "rank": rank, "direction": "most_correlated",
                "peer_id": peer, "peer_symbol": id2sym.get(peer, peer.upper()),
                "peer_name": id2name.get(peer, peer), "correlation": round(float(val), 6)
            })
        for rank, (peer, val) in enumerate(col.nsmallest(k).items(), 1):
            rows.append({
                "asset_id": asset, "asset_symbol": sym, "asset_name": name,
                "rank": rank, "direction": "least_correlated",
                "peer_id": peer, "peer_symbol": id2sym.get(peer, peer.upper()),
                "peer_name": id2name.get(peer, peer), "correlation": round(float(val), 6)
            })

    return pd.DataFrame(rows)


def compute_global_extreme_pairs(corr: pd.DataFrame, meta: pd.DataFrame, k: int = 20) -> pd.DataFrame:
    id2sym  = {str(row["id"]): str(row["symbol"]).upper() for _, row in meta.iterrows()}
    id2name = {str(row["id"]): str(row["name"])           for _, row in meta.iterrows()}
    cols = [str(c) for c in corr.columns]
    n    = len(cols)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            v = corr.iloc[i, j]
            if pd.isna(v): continue
            pairs.append((cols[i], cols[j], float(v)))

    if not pairs: return pd.DataFrame()
    pdf = pd.DataFrame(pairs, columns=["a", "b", "corr"])

    def enrich(df, direction):
        df = df.copy()
        df["direction"] = direction
        df["a_symbol"]  = df["a"].map(id2sym)
        df["a_name"]    = df["a"].map(id2name)
        df["b_symbol"]  = df["b"].map(id2sym)
        df["b_name"]    = df["b"].map(id2name)
        return df[["direction","a","a_symbol","a_name","b","b_symbol","b_name","corr"]]

    return pd.concat([
        enrich(pdf.nlargest(k,  "corr"), "most"),
        enrich(pdf.nsmallest(k, "corr"), "least"),
    ], ignore_index=True)

# =============================================================================
#  ՎԻԶՈՒԱԼԻԶԱՑԻԱ
# =============================================================================

def _sym(cid, id2sym):
    return id2sym.get(str(cid), str(cid)[:6].upper())

def plot_full_clustermap(corr, meta):
    n      = corr.shape[0]
    id2sym = {str(row["id"]): str(row["symbol"]).upper() for _, row in meta.iterrows()}
    labels = [_sym(c, id2sym) for c in corr.columns]
    fs     = max(4, 9 - n // 40)
    log.info(f"Գծագրվում է ամբողջական կլաստերային քարտեզը (clustermap) ({n}x{n}) ...")
    cm = sns.clustermap(
        corr, cmap="RdYlGn", center=0, vmin=-1, vmax=1, linewidths=0,
        xticklabels=labels, yticklabels=labels,
        figsize=(max(24, n * 0.33), max(22, n * 0.30)),
        cbar_kws={"shrink": 0.4, "label": "Պիրսոնի r"},
        dendrogram_ratio=0.10, annot=False,
    )
    plt.setp(cm.ax_heatmap.get_xticklabels(), rotation=90, fontsize=fs)
    plt.setp(cm.ax_heatmap.get_yticklabels(), rotation=0,  fontsize=fs)
    cm.ax_heatmap.set_title(
        f"{n} ակտիվ\nՕրական լոգ-եկամտաբերություն  |  {START_DATE} -> {END_DATE}",
        fontsize=13, fontweight="bold", pad=18,
    )
    path = RESULTS_DIR / "1_full_correlation_clustermap.png"
    cm.savefig(path, dpi=130, bbox_inches="tight")
    plt.close("all")
    log.info(f"Պահպանվեց => {path}")

def plot_market_avg_correlation(corr, meta):
    id2sym = {str(row["id"]): str(row["symbol"]).upper() for _, row in meta.iterrows()}
    n = corr.shape[0]
    avg = corr.apply(lambda col: col.drop(index=col.name, errors="ignore").mean()).sort_values(ascending=False)
    med = avg.median()
    colors = ["#27ae60" if v >= med else "#e74c3c" for v in avg.values]

    fig, ax = plt.subplots(figsize=(max(18, n * 0.20), 8))
    ax.bar([_sym(c, id2sym) for c in avg.index], avg.values, color=colors, width=0.8, alpha=0.85)
    ax.axhline(avg.mean(), color="#3498db", lw=2.0, ls="--", label=f"Շուկայի միջինը = {avg.mean():.4f}")
    ax.axhline(med, color="#9b59b6", lw=1.8, ls=":", label=f"Շուկայի մեդիանը = {med:.4f}")
    ax.set_xlabel("Ակտիվ", fontsize=10)
    ax.set_ylabel("Միջին Պիրսոնի r (մնացած բոլորի նկատմամբ)", fontsize=11)
    ax.set_title(f"Կանաչ - մեդիանից բարձր շուկայական ինտեգրում  |  Կարմիր - մեդիանից ցածր", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(avg)))
    ax.set_xticklabels([_sym(c, id2sym) for c in avg.index], rotation=90, fontsize=max(4, 8 - n // 60))
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = RESULTS_DIR / "2_market_avg_correlation_bar.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Պահպանվեց => {path}")
    
def plot_global_extreme_pairs(pairs_df, k=20):
    if pairs_df.empty: return
    most  = pairs_df[pairs_df["direction"] == "most"].copy()
    least = pairs_df[pairs_df["direction"] == "least"].copy()

    def plabel(row): return f"{row['a_symbol']} <-> {row['b_symbol']}"
    fig, axes = plt.subplots(1, 2, figsize=(20, max(8, k * 0.60)))
    fig.suptitle(f"Գլոբալ Ամենաշատ և Ամենաքիչ Կորելացված Ակտիվների Զույգերը  |  {START_DATE} -> {END_DATE}", fontsize=14, fontweight="bold", y=1.01)

    for ax, df, color, title in [(axes[0], most, "#27ae60", f"Լավագույն {k} ԱՄԵՆԱՇԱՏ Կորելացված Զույգերը"), (axes[1], least, "#c0392b", f"Լավագույն {k} ԱՄԵՆԱՔԻՉ Կորելացված Զույգերը")]:
        labels = [plabel(r) for _, r in df.iterrows()]
        vals   = df["corr"].values
        bars   = ax.barh(labels, vals, color=color, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=8)
        ax.set_xlim(-1.1, 1.1)
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Պիրսոնի r", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = RESULTS_DIR / "3_global_extreme_pairs.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Պահպանվեց => {path}")

def plot_correlation_distribution(corr):
    flat = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().values
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.hist(flat, bins=80, color="#3498db", alpha=0.70, edgecolor="white", density=True, label="Հիստոգրամ")
    try:
        kde_fn = gaussian_kde(flat)
        xs = np.linspace(-1, 1, 400)
        ax.plot(xs, kde_fn(xs), color="#c0392b", lw=2.5, label="Խտության գնահատում (KDE)")
    except Exception: pass
    ax.axvline(flat.mean(), color="#2ecc71", lw=2.0, ls="--", label=f"Միջին   = {flat.mean():.4f}")
    ax.axvline(float(np.median(flat)), color="#e67e22", lw=2.0, ls=":", label=f"Մեդիան = {float(np.median(flat)):.4f}")
    ax.set_xlabel("Պիրսոնի Կորելացիայի Գործակից", fontsize=11)
    ax.set_ylabel("Խտություն", fontsize=11)
    ax.set_title(f"Բոլոր Զույգային Կորելացիաների Բաշխումը\nn = {len(flat):,} եզակի զույգ  |  {START_DATE} -> {END_DATE}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    path = RESULTS_DIR / "4_full_correlation_distribution.png"
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Պահպանվեց => {path}")
def plot_per_asset_top_k_grid(all_top_k, meta, max_per_fig=12):
    if all_top_k.empty: return
    id2sym = {str(row["id"]): str(row["symbol"]).upper() for _, row in meta.iterrows()}
    assets = all_top_k["asset_id"].unique().tolist()
    chunks = [assets[i:i + max_per_fig] for i in range(0, len(assets), max_per_fig)]
    log.info(f"Գծագրվում է ըստ ակտիվի լավագույն K-երի ցանցը ({len(assets)} ակտիվ, {len(chunks)} նկար) ...")

    for fig_idx, chunk in enumerate(chunks, 1):
        n_a  = len(chunk)
        ncol = min(4, n_a)
        nrow = (n_a + ncol - 1) // ncol
        fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 5.5, nrow * 4.2))
        axes_flat = np.array(axes).flatten() if isinstance(axes, np.ndarray) else [axes]

        for i, asset in enumerate(chunk):
            ax  = axes_flat[i]
            sub = all_top_k[all_top_k["asset_id"] == asset]
            most  = sub[sub["direction"] == "most_correlated"].sort_values("rank")
            least = sub[sub["direction"] == "least_correlated"].sort_values("rank")

            m_lbl  = list(most["peer_symbol"])
            l_lbl  = list(least["peer_symbol"])
            labels = m_lbl + l_lbl
            vals   = list(most["correlation"]) + list(least["correlation"])
            colors = ["#27ae60"] * len(m_lbl) + ["#c0392b"] * len(l_lbl)

            bars = ax.barh(labels, vals, color=colors, alpha=0.85, edgecolor="white")
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=7)
            ax.axvline(0, color="black", lw=0.6, ls="--")
            if len(m_lbl) > 0: ax.axhline(len(m_lbl) - 0.5, color="gray", lw=1.2, ls=":")
            ax.set_xlim(-1.05, 1.05)
            ax.set_title(id2sym.get(asset, asset.upper()), fontsize=10, fontweight="bold")
            ax.tick_params(axis="y", labelsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="x", alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(handles=[Patch(color="#27ae60", label=f"Լավագույն {TOP_K} ամենաշատ կորել."), Patch(color="#c0392b", label=f"Լավագույն {TOP_K} ամենաքիչ կորել.")], fontsize=7, loc="lower right")

        for j in range(n_a, len(axes_flat)): axes_flat[j].set_visible(False)
        fig.suptitle(f"Յուրաքանչյուր Ակտիվի Լավագույն {TOP_K} Ամենաշատ և Ամենաքիչ Կորելացված Ակտիվները  (Նկար {fig_idx}/{len(chunks)})  |  {START_DATE} -> {END_DATE}", fontsize=12, fontweight="bold", y=1.005)
        plt.tight_layout()
        path = RESULTS_DIR / f"5_per_asset_top_k_grid_{fig_idx:03d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Պահպանվեց => {path}")
def plot_extreme_market_integration(avg_df, k=5):
    if avg_df.empty or len(avg_df) < k * 2:
        return
        
    top_k = avg_df.head(k).copy()
    bot_k = avg_df.tail(k).copy()
    
    plot_df = pd.concat([bot_k, top_k]).sort_values('mean_corr_with_market', ascending=True)
    colors = ['#c0392b' if x in bot_k['symbol'].values else '#27ae60' for x in plot_df['symbol']]
    labels = [f"{row['symbol']} ({row['name']})" for _, row in plot_df.iterrows()]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, plot_df['mean_corr_with_market'], color=colors, alpha=0.85, edgecolor="white")
    ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=10, fontweight="bold")
    
    ax.set_xlabel("Միջին Պիրսոնի r (մնացած բոլոր ակտիվների նկատմամբ)", fontsize=11)
    ax.set_title(f"Լավագույն {k} Ամենաշատ և Ամենաքիչ Շուկայական Ինտեգրում Ունեցող Ակտիվները\nԿանաչ = Շուկայի Պրոքսիներ (Բազային)  |  Կարմիր = Անկախ Ակտիվներ (Շեղումներ)", fontsize=13, fontweight="bold", pad=15)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.axhline(k - 0.5, color="gray", lw=1.5, ls=":")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    
    path = RESULTS_DIR / "6_extreme_market_integration.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Պահպանվեց => {path}")



def export_all(corr, all_top_k, global_pairs, meta):
    id2name = {str(row["id"]): str(row["name"])           for _, row in meta.iterrows()}
    id2sym  = {str(row["id"]): str(row["symbol"]).upper() for _, row in meta.iterrows()}

    corr.to_csv(RESULTS_DIR / "corr_matrix_full.csv")
    corr.to_parquet(RESULTS_DIR / "corr_matrix_full.parquet")
    log.info("Պահպանվեց. corr_matrix_full.csv / .parquet")

    all_top_k.to_csv(RESULTS_DIR / "all_assets_top_k_corressions.csv", index=False)
    log.info("Պահպանվեց. all_assets_top_k_correlations.csv")

    global_pairs.to_csv(RESULTS_DIR / "global_extreme_pairs.csv", index=False)
    log.info("Պահպանվեց. global_extreme_pairs.csv")

    for asset, grp in all_top_k.groupby("asset_id"):
        sym   = id2sym.get(str(asset), str(asset)).upper()
        fname = f"{sym}_{asset}.csv".replace("/", "_")
        grp.to_csv(PER_ASSET_DIR / fname, index=False)
    log.info(f"Յուրաքանչյուր ակտիվի CSV ֆայլերը պահպանվեցին ({all_top_k['asset_id'].nunique()} ֆայլ) => {PER_ASSET_DIR}/")

    avg = corr.apply(lambda col: col.drop(index=col.name, errors="ignore").mean())
    avg_df = pd.DataFrame({
        "coin_id": avg.index,
        "symbol":  [id2sym.get(str(c), str(c)) for c in avg.index],
        "name":    [id2name.get(str(c), str(c)) for c in avg.index],
        "mean_corr_with_market": avg.values,
    }).sort_values("mean_corr_with_market", ascending=False)
    avg_df.to_csv(RESULTS_DIR / "market_avg_correlation_per_asset.csv", index=False)
    log.info("Պահպանվեց. market_avg_correlation_per_asset.csv")

    n    = corr.shape[0]
    flat = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack().values
    
    mp = global_pairs[global_pairs["direction"] == "most"].iloc[0] if not global_pairs.empty else None
    lp = global_pairs[global_pairs["direction"] == "least"].iloc[0] if not global_pairs.empty else None

    # Top 5 Most and Least Market Integrated
    top_5_market = avg_df.head(5)
    bot_5_market = avg_df.tail(5)

    report_top_5 = "\n".join([f"  {row['symbol']:<5} ({row['name']:<15}) : միջին r = {row['mean_corr_with_market']:.4f}" for _, row in top_5_market.iterrows()])
    report_bot_5 = "\n".join([f"  {row['symbol']:<5} ({row['name']:<15}) : միջին r = {row['mean_corr_with_market']:.4f}" for _, row in bot_5_market.iterrows()])

    report = f"""
=====================================================================
   ԿՐԻՊՏՈԱՐԺՈՒՅԹՆԵՐԻ ՇՈՒԿԱ - ԱՄԲՈՂՋԱԿԱՆ ԿՈՐԵԼԱՑԻՈՆ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ | ԱՄՓՈՓՈՒՄ
=====================================================================

Վերլուծության Պարամետրեր
------------------------
  Ժամանակահատված    : {START_DATE}  ->  {END_DATE}
  Սկզբնական թիվ     : լավագույն {TOP_N} ակտիվ
  Վերջնական ցանկ    : {n} ակտիվ (բոլոր ֆիլտրերից հետո)
  Նվազագույն ծածկ.  : Առևտրային օրերի {MIN_DATA_COVERAGE*100:.0f}%-ը
  Սթեյբլքոինի շեղում: {STABLECOIN_STD_THRESH*100:.1f}%/օր (ֆիլտրի շեմ)
  Լավագույն K ըստ ակտ.: {TOP_K} ամենաշատ + {TOP_K} ամենաքիչ կորելացված

Բոլոր Զույգերի Վիճակագրություն
------------------------------
  Ընդհանուր եզակի զույգեր: {len(flat):,}
  Միջին  կորելացիա       : {flat.mean():.4f}
  Մեդիան կորելացիա       : {float(np.median(flat)):.4f}
  Կորել. ստանդարտ շեղում : {flat.std():.4f}
  Նվազ.  կորելացիա       : {flat.min():.4f}
  Առավել. կորելացիա      : {flat.max():.4f}
"""
    if mp is not None and lp is not None:
        report += f"""
Ամենաշատ Կորելացված Զույգ 
----------------------------------
  {mp['a_symbol']}  ({mp['a_name']})
  <->  {mp['b_symbol']}  ({mp['b_name']})
  r = {mp['corr']:.4f}

Ամենաքիչ Կորելացված Զույգ 
----------------------------------
  {lp['a_symbol']}  ({lp['a_name']})
  <->  {lp['b_symbol']}  ({lp['b_name']})
  r = {lp['corr']:.4f}
"""
    report += f"""
5 Ամենաշատ Շուկայական Ինտեգրում Ունեցող Ակտիվներ (Պրոքսի Թեկնածուներ)
-------------------------------------------------------------------------------------
{report_top_5}

5 Ամենաքիչ Շուկայական Ինտեգրում Ունեցող Ակտիվներ (Outlier Թեկնածուներ)
-----------------------------------------------------------------------------
{report_bot_5}
"""
    report += f"\nՍտեղծվել է : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

    (RESULTS_DIR / "summary_report.txt").write_text(report, encoding="utf-8")
    log.info("Պահպանվեց. summary_report.txt")
    print(report)




def main():
    log.info("=" * 68)
    log.info("   ԿՐԻՊՏՈԱՐԺՈՒՅԹՆԵՐԻ ԱՄԲՈՂՋԱԿԱՆ ՇՈՒԿԱՅԻ ԿՈՐԵԼԱՑԻՈՆ ՎԵՐԼՈՒԾՈՒԹՅՈՒՆ — ՍԿԻԶԲ")
    log.info(f"   Ժամանակահատված : {START_DATE}  ->  {END_DATE}")
    log.info(f"   Լավագույն-N : {TOP_N}    Նվազագույն-ծածկույթ : {MIN_DATA_COVERAGE*100:.0f}%    K : {TOP_K}")
    log.info("=" * 68)

    df = get_top_n_coins(TOP_N)
    log.info(f"Բեռնվել է {len(df)} մետաղադրամ")

    df = f1_remove_known_stablecoins(df)
    df = f2_remove_low_mcap(df)

    price_mat = f3_build_price_matrix(df)

    if price_mat.empty:
        log.error("Գների մատրիցը դատարկ է։ Կատարումը դադարեցվում է։")
        return

    price_mat = f4_filter_coverage(price_mat)
    price_mat = f5_filter_stat_stablecoins(price_mat)

    remaining = set(str(c) for c in price_mat.columns)
    df = df[df["id"].isin(remaining)].copy().reset_index(drop=True)
    log.info(f"Վերջնական ցանկ. {len(df)} ակտիվ")

    if len(df) == 0:
        log.error("Ֆիլտրումից հետո ոչ մի ակտիվ չի մնացել։ Ստուգեք ֆիլտրերը կամ տվյալների հասանելիությունը։")
        return

    lr = compute_log_returns(price_mat)
    log.info(f"Լոգ-եկամտաբերության մատրից. {lr.shape}")

    corr = compute_correlation_matrix(lr)
    log.info(f"Կորելացիոն մատրից. {corr.shape[0]}x{corr.shape[1]}")

    if corr.shape[0] == 0:
        log.error("Կորելացիոն մատրիցը դատարկ է։ Հնարավոր չէ շարունակել։")
        return

    all_top_k = compute_all_top_k(corr, df, k=TOP_K)
    global_pairs = compute_global_extreme_pairs(corr, df, k=20)
    
    avg = corr.apply(lambda col: col.drop(index=col.name, errors="ignore").mean())
    id2name = {str(row["id"]): str(row["name"]) for _, row in df.iterrows()}
    id2sym  = {str(row["id"]): str(row["symbol"]).upper() for _, row in df.iterrows()}
    avg_df = pd.DataFrame({
        "coin_id": avg.index,
        "symbol":  [id2sym.get(str(c), str(c)) for c in avg.index],
        "name":    [id2name.get(str(c), str(c)) for c in avg.index],
        "mean_corr_with_market": avg.values,
    }).sort_values("mean_corr_with_market", ascending=False)

    plot_full_clustermap(corr, df)
    plot_market_avg_correlation(corr, df)
    plot_global_extreme_pairs(global_pairs, k=20)
    plot_correlation_distribution(corr)
    plot_per_asset_top_k_grid(all_top_k, df, max_per_fig=12)
    plot_extreme_market_integration(avg_df, k=5) 

    export_all(corr, all_top_k, global_pairs, df)

    log.info("=" * 68)
    log.info("   ՎԵՐԼՈՒԾՈՒԹՅՈՒՆՆ ԱՎԱՐՏՎԱԾ Է")
    log.info(f"   Արդյունքներ => {RESULTS_DIR.resolve()}")
    log.info("=" * 68)


if __name__ == "__main__":
    main()


# 2

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.filterwarnings("ignore")

start_date = '2018-01-01'
end_date = '2026-01-01'

# ԱԿՏԻՎՆԵՐԻ ԽԻՍՏ ՀԵՐԹԱԿԱՆՈՒԹՅՈՒՆԸ
desired_order = ['S&P500', 'GOLD', 'DXY', 'US10Y', 'ETH', 'BTC', 'QNT']

tickers = {
    'S&P500': '^GSPC',
    'GOLD': 'GC=F',
    'DXY': 'DX-Y.NYB',
    'US10Y': '^TNX',
    'ETH': 'ETH-USD',
    'BTC': 'BTC-USD',
    'QNT': 'QNT-USD'
}

print("Ներբեռնվում են 7 հիմնական ակտիվների տվյալները...")
raw_data = yf.download(list(tickers.values()), start=start_date, end=end_date, progress=False)['Close']
raw_data.rename(columns={v: k for k, v in tickers.items()}, inplace=True)

# 2. Տվյալների սինխրոնիզացիա (Inner Join մեթոդաբանություն)
aligned_prices = raw_data.dropna()

# 3. Եկամտաբերությունների հաշվարկ
# Սկզբից հաշվում ենք լոգ-եկամտաբերություն բոլորի համար
log_returns = np.log(aligned_prices / aligned_prices.shift(1)) * 100
# ԱՊԱ US10Y-ը. վերագրում ենք միայն բացարձակ տարբերությունը
log_returns['US10Y'] = aligned_prices['US10Y'].diff()
# Մաքրում ենք առաջին դատարկ տողը
log_returns = log_returns.dropna()

print(f"Inner Join-ից հետո մնացած աշխատանքային օրերի քանակը: {len(log_returns)}\n")

# 4. Նկարագրական Վիճակագրության Հաշվարկ
stats_list = []
for col in log_returns.columns:
    series = log_returns[col]
    adf_result = adfuller(series.dropna())
    stats_list.append({
        'Ակտիվ': col,
        'Միջին (%)': series.mean(),
        'Ստ. Շեղում (%)': series.std(),
        'Շեղվածություն (Skewness)': series.skew(),
        'Էքսցես (Kurtosis)': series.kurtosis(),
        'Մինիմում (%)': series.min(),
        'Մաքսիմում (%)': series.max(),
        'ADF p-value': adf_result[1] 
    })

desc_stats_df = pd.DataFrame(stats_list).set_index('Ակտիվ')
desc_stats_df = desc_stats_df.round(4)

# --- ԽԻՍՏ ՀԵՐԹԱԿԱՆՈՒԹՅԱՆ ԱՊԱՀՈՎՈՒՄ ---
desc_stats_df = desc_stats_df.loc[desired_order]

desc_stats_df.to_csv('Նկարագրական_Վիճակագրություն.csv', sep=';', decimal=',', encoding='utf-8-sig')

print("--- ՆԿԱՐԱԳՐԱԿԱՆ ՎԻՃԱԿԱԳՐՈՒԹՅՈՒՆ ---")
print(desc_stats_df.to_string())
print("\nԱղյուսակը պահպանվել է 'Նկարագրական_Վիճակագրություն.csv' ֆայլում:")

# 5. Ռեժիմային փուլերով (Structural Breaks) Վիզուալիզացիա
plt.figure(figsize=(16, 10))
for i, col in enumerate(['BTC', 'S&P500', 'GOLD']):
    plt.subplot(3, 1, i+1)
    plt.plot(log_returns.index, log_returns[col], color='tab:blue' if i==0 else ('tab:green' if i==1 else 'tab:orange'), linewidth=0.8)
    
    plt.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', linewidth=1.5, label='COVID-19 Շոկ (Մարտ 2020)')
    plt.axvline(pd.to_datetime('2024-01-10'), color='purple', linestyle='--', linewidth=1.5, label='BTC Spot ETF (Հունվար 2024)')
    
    plt.title(f'{col} - Օրական Եկամտաբերություններ', fontsize=12, fontweight='bold')
    plt.ylabel('Եկամտաբերություն (%)')
    if i == 0:
        plt.legend(loc='upper right')

plt.tight_layout()
plt.savefig('Ռեժիմային_Խզումներ_Վոլատիլություն.png', dpi=150)
plt.show()

# Աշխատանքային ֆայլը պահպանում ենք
log_returns = log_returns[desired_order]
log_returns.to_csv('Աշխատանքային_Եկամտաբերություններ.csv', sep=';', encoding='utf-8-sig')


#3


import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen


prices_for_johansen = aligned_prices.copy()

# Լոգարիթմում ենք բոլորը, ԲԱՑԻ US10Y-ից (տոկոսադրույքը մնում է բացարձակ մակարդակով)
cols_to_log = ['S&P500', 'GOLD', 'DXY', 'ETH', 'BTC', 'QNT']
for col in cols_to_log:
    prices_for_johansen[col] = np.log(prices_for_johansen[col])

prices_for_johansen = prices_for_johansen.dropna()

# --- ճիշտ հերթականությունը ---
desired_order = ['S&P500', 'GOLD', 'DXY', 'US10Y', 'ETH', 'BTC', 'QNT']
prices_for_johansen = prices_for_johansen[desired_order]

# 2. Johansen թեստի ֆունկցիան
def johansen_on_prices(df):
    # det_order=0 նշանակում է մոդելում առկա է միայն հաստատուն (constant)
    # k_ar_diff=1 նշանակում է 1 հապաղում տարբերություններում (համապատասխանում է VAR(1)-ին)
    res = coint_johansen(df, det_order=0, k_ar_diff=1)
    
    # Ձևավորում ենք աղյուսակը
    output = pd.DataFrame({
        'Trace Stat': res.lr1, 
        'Crit Value (5%)': res.cvt[:, 1]
    }, index=[f"r <= {i} (Rank {i})" for i in range(df.shape[1])])
    
    # Եթե Trace > Critical Value, ուրեմն կոինտեգրացիա կա
    output['Significant (Cointegration)'] = output['Trace Stat'] > output['Crit Value (5%)']
    
    return output

print("\n--- ՅՈՀԱՆՍԵՆԻ ԿՈԻՆՏԵԳՐԱՑԻՈՆ ԹԵՍՏ (TRACE STATISTIC) ---")
johansen_results = johansen_on_prices(prices_for_johansen)
print(johansen_results.round(4))

# Ստուգում ենք ամենաառաջին վարկածը (r = 0, այսինքն՝ արդյո՞ք կա գոնե 1 կոինտեգրացիոն կապ)
r_0_trace = johansen_results.iloc[0]['Trace Stat']
r_0_crit = johansen_results.iloc[0]['Crit Value (5%)']

print("\n--- ԵԶՐԱԿԱՑՈՒԹՅՈՒՆ ---")
if r_0_trace < r_0_crit:
    print(f"Trace վիճակագրությունը ({r_0_trace:.4f}) ՓՈՔՐ Է 5% կրիտիկական արժեքից ({r_0_crit:.4f}):")
    print("Ակտիվների միջև ԵՐԿԱՐԱԺԱՄԿԵՏ ԿՈԻՆՏԵԳՐԱՑԻՈՆ ԿԱՊԸ ԲԱՑԱԿԱՅՈՒՄ Է:")
    print("Վեկտորային ավտոռեգրեսիոն (VAR) մոդելի կիրառումը եկամտաբերությունների նկատմամբ ԼԻՈՎԻՆ ՀԻՄՆԱՎՈՐՎԱԾ Է:")
else:
    print("Ակտիվների միջև առկա է կոինտեգրացիա (VECM մոդելի կիրառման անհրաժեշտություն կա):")


#4


from statsmodels.tsa.api import VAR


print("=== VAR մոդելի օպտիմալ հապաղումների ընտրություն ===")
model = VAR(data)

lag_selection = model.select_order(maxlags=8)

print(lag_selection.summary())

optimal_lag = lag_selection.aic
print(f"\nԱվտոմատ ընտրված օպտիմալ լագը ըստ AIC չափանիշի: {optimal_lag}")


#5


import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
from statsmodels.stats.diagnostic import het_arch
import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('ՃՇԳՐՏՎԱԾ_Եկամտաբերություններ.csv', index_col=0, parse_dates=True, sep=';')
model = VAR(data)
results = model.fit(1)

# 2. Portmanteau (Q-stat) Test (Ավտոկորելացիայի ստուգում ողջ համակարգում)
white_test = results.test_whiteness(nlags=10)
print(f"2. Portmanteau Test (p-value): {white_test.pvalue:.4e}")
if white_test.pvalue < 0.05:
    print("   -> Եզրակացություն: Առկա է ավտոկորելացիա (Մերժվում է H0):")

# 3. ARCH-LM Test (Հետերոսկեդաստիկության ստուգում յուրաքանչյուր ակտիվի համար)
print("\n3. ARCH-LM Test յուրաքանչյուր շարքի մնացորդի համար (Lag=5):")
residuals = results.resid
for col in residuals.columns:
    arch_res = het_arch(residuals[col], nlags=5)
    print(f"   - {col: <7}: p-value = {arch_res[1]:.4e}")
    
    
eigenvalues = np.linalg.eigvals(results.coefs[0])
print(np.abs(eigenvalues))


#6


import numpy as np
from statsmodels.tsa.api import VAR

# հաշվում ենք մոդելը Lag=1-ով, որպեսզի բոլոր փոփոխականները տեղում լինեն
model = VAR(data)
results = model.fit(1)
Sigma = results.sigma_u.values
K = results.neqs
sigma_jj = np.diag(Sigma)

# Հաշվում ենք TSI-ը H=5 H=10 և H=15 համար
for h_val in [5,10, 15]:
    A_h = results.ma_rep(maxn=h_val) 
    num_h = np.zeros((K, K))
    den_h = np.zeros(K)
    
    for h in range(h_val):
        num_h += (A_h[h] @ Sigma)**2 / sigma_jj
        den_h += np.diag(A_h[h] @ Sigma @ A_h[h].T)
        
    theta_h = num_h / den_h[:, None]
    tsi_val = float(((theta_h / theta_h.sum(axis=1)[:, None] * 100).sum() - np.diag((theta_h / theta_h.sum(axis=1)[:, None] * 100)).sum()) / K)
    print(f"H={h_val} դեպքում TSI: {tsi_val:.2f}%")


#7


import pandas as pd
import numpy as np
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# 1. Բեռնում ենք տվյալները
data = pd.read_csv('ՃՇԳՐՏՎԱԾ_Եկամտաբերություններ.csv', index_col=0, parse_dates=True, sep=';')

# 2. VAR Մոդելի կառուցում (Lag = 1)
lag_optimal = 1
horizon = 10
model = VAR(data)
results = model.fit(lag_optimal)

Sigma = results.sigma_u.values
K = results.neqs
A = results.ma_rep(maxn=horizon)
sigma_jj = np.diag(Sigma)

num = np.zeros((K, K))
den = np.zeros(K)
for h in range(horizon):
    num += (A[h] @ Sigma)**2 / sigma_jj
    den += np.diag(A[h] @ Sigma @ A[h].T)
    
theta = num / den[:, None]
theta_tilde = (theta / theta.sum(axis=1)[:, None]) * 100

# 3. ՄԱՏՐԻՑԻ ՏՊՈՒՄ
cols = data.columns
matrix_df = pd.DataFrame(theta_tilde, index=cols, columns=cols)

from_others = theta_tilde.sum(axis=1) - np.diag(theta_tilde)
to_others = theta_tilde.sum(axis=0) - np.diag(theta_tilde)

# Ավելացնում ենք "FROM Others" սյունակը
matrix_df['FROM Others'] = from_others

# Ավելացնում ենք "TO Others" տողը (ներառյալ ընդհանուր գումարը աջ անկյունում)
to_others_row = list(to_others) + [sum(to_others)]
matrix_df.loc['TO Others'] = to_others_row

# Հաշվում ենք NET (Զուտ) փոխանցումները միայն 7 ակտիվների համար
matrix_df.loc['NET'] = pd.Series(to_others - from_others, index=cols)

tsi = sum(to_others) / K

print("=== ԴԻԲՈԼԴ-ՅԻԼՄԱԶԻ ՎԵՐՋՆԱԿԱՆ ՄԱՏՐԻՑ (Lag=1) ===")
print(matrix_df.round(2).fillna(''))
print(f"\nՎերջնական TSI: {tsi:.2f}%\n")


# 4. ROLLING WINDOW TSI (150 օր)
window = 150
rolling_tsi = []
dates = []

print("Հաշվարկվում է Rolling TSI-ը (կտևի մի քանի վայրկյան)...")
for i in range(window, len(data)):
    window_data = data.iloc[i-window:i]
    try:
        res_w = VAR(window_data).fit(lag_optimal)
        Sigma_w = res_w.sigma_u.values
        A_w = res_w.ma_rep(maxn=horizon)
        sigma_jj_w = np.diag(Sigma_w)
        
        num_w = np.zeros((K, K))
        den_w = np.zeros(K)
        for h in range(horizon):
            num_w += (A_w[h] @ Sigma_w)**2 / sigma_jj_w
            den_w += np.diag(A_w[h] @ Sigma_w @ A_w[h].T)
            
        theta_w = num_w / den_w[:, None]
        theta_tilde_w = (theta_w / theta_w.sum(axis=1)[:, None]) * 100
        tsi_w = float((theta_tilde_w.sum() - np.diag(theta_tilde_w).sum()) / K)
        
        rolling_tsi.append(tsi_w)
        dates.append(data.index[i-1])
    except:
        pass

# Գրաֆիկի կառուցում և պահպանում
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(dates, rolling_tsi, color='darkred', linewidth=1.5)
plt.title('Total Volatility Spillover Index (Rolling 150-day window, Lag=1)', fontsize=14, fontweight='bold')
plt.ylabel('TSI (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('Rolling_TSI_Lag1.png', dpi=300)
print("\nԳրաֆիկը հաջողությամբ պահպանվել է որպես 'Rolling_TSI_Lag1.png':")


#8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

# --- ԱԿՏԻՎՆԵՐԻ ՀԵՐԹԱԿԱՆՈՒԹՅԱՆ ՍԱՀՄԱՆՈՒՄ ---
desired_order = ['S&P500', 'GOLD', 'DXY', 'US10Y', 'ETH', 'BTC', 'QNT']

# 1. Տվյալների ներբեռնում
print("Ներբեռնվում են աշխատանքային տվյալները...")
data = pd.read_csv('Աշխատանքային_Եկամտաբերություններ.csv', index_col=0, parse_dates=True, sep=';')

# եթե թվերի մեջ ստորակետ է մնացել, դարձնում ենք կետ և վերածում float-ի
data = data.replace(',', '.', regex=True).astype(float)
data = data.dropna()

data = data[desired_order]

print(f"Հաջողությամբ բեռնվեց {data.shape[1]} ակտիվ և {data.shape[0]} աշխատանքային օր։")

# 2. VAR մոդելի լագի ընտրություն
print("\n--- VAR ՄՈԴԵԼԻ ԼԱԳԻ ԸՆՏՐՈՒԹՅՈՒՆ ---")
model = VAR(data)
lag_selection = model.select_order(maxlags=10)

optimal_lag = lag_selection.aic
if optimal_lag == 0:
    print("AIC-ն ընտրել է 0: Սահմանվում է լագ = 1՝ շոկերի փոխանցումը ֆիքսելու համար:")
    optimal_lag = 1
else:
    print(f"Ընտրված օպտիմալ լագը ըստ AIC-ի: {optimal_lag}")

# 3. VAR մոդելի գնահատում
var_fitted = model.fit(optimal_lag)

# 4. ԳԵՆԵՐԱԼԻԶԱՑՎԱԾ (Generalized) FEVD-Ի ՀԱՇՎԱՐԿ
def generalized_fevd(var_results, horizon):
    K = var_results.neqs
    Sigma = var_results.sigma_u.values if hasattr(var_results.sigma_u, 'values') else var_results.sigma_u
    sigma_jj = np.diag(Sigma)
    
    A = var_results.ma_rep(maxn=horizon-1) 
    
    numerator = np.zeros((K, K))
    denominator = np.zeros(K)
    
    for h in range(horizon):
        Ah = A[h]
        num_h = (Ah @ Sigma)**2 / sigma_jj
        numerator += num_h
        
        den_h = np.diag(Ah @ Sigma @ Ah.T)
        denominator += den_h
        
    theta = numerator / denominator[:, None]
    theta_tilde = (theta / theta.sum(axis=1)[:, None]) * 100
    return theta_tilde

forecast_horizon = 10
decomp_matrix = generalized_fevd(var_fitted, forecast_horizon)
assets = data.columns

# 5. Spillover Աղյուսակի կառուցում
spillover_df = pd.DataFrame(decomp_matrix, index=assets, columns=assets)

own_variance = np.diag(spillover_df)
spillover_df['FROM Others'] = spillover_df.sum(axis=1) - own_variance
spillover_df.loc['TO Others'] = spillover_df[assets].sum(axis=0) - own_variance

total_spillover = spillover_df.loc[assets, 'FROM Others'].sum() / len(assets)
spillover_df.loc['TO Others', 'FROM Others'] = total_spillover

net_spillover_row = spillover_df.loc['TO Others', assets] - spillover_df.loc[assets, 'FROM Others'].values
spillover_df.loc['NET'] = net_spillover_row
spillover_df.loc['NET', 'FROM Others'] = np.nan # Այս վանդակը պետք է դատարկ մնա

spillover_df = spillover_df.round(2)

print("\n--- DIEBOLD-YILMAZ (2012) GENERALIZED SPILLOVER ԱՂՅՈՒՍԱԿ (%) ---")
print(spillover_df)
spillover_df.to_csv('Diebold_Yilmaz_Spillover_Table.csv', sep=';', decimal=',', encoding='utf-8-sig')
print("\nԱղյուսակը պահպանվել է 'Diebold_Yilmaz_Spillover_Table.csv' ֆայլում:")

# 6. Վիզուալիզացիա (Net Spillovers Heatmap)
net_spillover_matrix = pd.DataFrame(0.0, index=assets, columns=assets)
for i in assets:
    for j in assets:
        if i != j:
            # Զուտ շոկ i-ից դեպի j = (j-ն ստանում է i-ից) - (i-ն ստանում է j-ից)
            net_spillover_matrix.loc[i, j] = spillover_df.loc[j, i] - spillover_df.loc[i, j]

plt.figure(figsize=(12, 10))
sns.heatmap(net_spillover_matrix, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f', linewidths=0.5, 
            annot_kws={"size": 10})

# --- ԱՌԱՆՑՔՆԵՐԸ ԵՎ ՎԵՐՆԱԳԻՐԸ ---
plt.title('Զուտ Տատանողականության Փոխանցման Ջերմային Քարտեզ (Net Pairwise Spillovers)', fontsize=14, fontweight='bold', pad=20)
plt.ylabel('Շոկը փոխանցող (Transmitter)', fontsize=12, fontweight='bold')
plt.xlabel('Շոկը ստացող (Receiver)', fontsize=12, fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

# Պահպանում ենք ակադեմիական որակով (dpi=300)
plt.savefig('Net_Spillover_Matrix.png', dpi=300)
plt.show()

print(f"\nՀամակարգի Ընդհանուր Spillover Ինդեքսը (TSI) կազմում է՝ {total_spillover:.2f}%")


#9


from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

print("--- ԲԱՇԽՈՒՄՆԵՐԻ ԹԵՍՏԱՎՈՐՈՒՄ (AIC և BIC) ---")

for col in data.columns:
    best_aic = float('inf')
    best_bic = float('inf')
    dist_aic = ''
    dist_bic = ''
    
    for dist_type in ['normal', 't', 'ged']:
        try:
            am = arch_model(data[col], p=1, q=1, vol='Garch', dist=dist_type, rescale=False)
            res = am.fit(disp='off')
            
            if res.aic < best_aic:
                best_aic = res.aic
                dist_aic = dist_type
                
            if res.bic < best_bic:
                best_bic = res.bic
                dist_bic = dist_type
        except:
            continue
            
    print(f"{col}: Ըստ AIC -> {dist_aic}, Ըստ BIC -> {dist_bic}")


#10


import pandas as pd
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

results_dict = {}

# Ճիշտ բաշխումները՝ ըստ մեր AIC/BIC թեստերի հաղթողների
dist_map = {
    'S&P500': 't',
    'GOLD': 'ged',
    'DXY': 'ged',
    'US10Y': 't',
    'ETH': 't',
    'BTC': 't',
    'QNT': 't'
}

for col in data.columns:
    dist_type = dist_map[col]
    
    # Կառուցում ենք GARCH մոդելը ճիշտ բաշխմամբ
    am = arch_model(data[col], p=1, q=1, vol='Garch', dist=dist_type, rescale=False)
    res = am.fit(disp='off')
    
    omega = res.params.get('omega', 0)
    alpha = res.params.get('alpha[1]', 0)
    beta = res.params.get('beta[1]', 0)
    persistence = alpha + beta
    
    results_dict[col] = {
        'Omega': round(omega, 4), 
        'Alpha': round(alpha, 4), 
        'Beta': round(beta, 4), 
        'Persistence': round(persistence, 4)
    }

df_garch_final = pd.DataFrame(results_dict).T
print("--- ՎԵՐՋՆԱԿԱՆ ԵՎ ՃՇԳՐԻՏ ԱՂՅՈՒՍԱԿ 3.3.1 ---")
print(df_garch_final)


#11


import numpy as np
import pandas as pd
from arch import arch_model
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# ՔԱՅԼ 1. ՍՏԱՆԴԱՐՏԱՑՎԱԾ ՄՆԱՑՈՐԴՆԵՐԻ (z_t) ՍՏԱՑՈՒՄ ՃԻՇՏ ԲԱՇԽՈՒՄՆԵՐՈՎ
# -----------------------------------------------------------------------------

# Ամրագրում ենք մեր ապացուցած ճշգրիտ բաշխումները
dist_map = {
    'S&P500': 't',
    'GOLD': 'ged',
    'DXY': 'ged',
    'US10Y': 't',
    'ETH': 't',
    'BTC': 't',
    'QNT': 't'
}

# Ստուգում ենք և պահպանում տվյալների հաջորդականությունը
assets = ['S&P500', 'GOLD', 'DXY', 'US10Y', 'ETH', 'BTC', 'QNT']
data = data[assets]

std_resid = pd.DataFrame(index=data.index, columns=assets)
conditional_vols = pd.DataFrame(index=data.index, columns=assets)

print("Գնահատվում են միաչափ GARCH մոդելները...")
for col in assets:
    am = arch_model(data[col], p=1, q=1, vol='Garch', dist=dist_map[col], rescale=False)
    res = am.fit(disp='off')
    
    # Պահպանում ենք ստանդարտացված մնացորդները (z_t = e_t / sigma_t)
    std_resid[col] = res.resid / res.conditional_volatility
    conditional_vols[col] = res.conditional_volatility

# -----------------------------------------------------------------------------
# ՔԱՅԼ 2. DCC-GARCH(1,1) ՄՈԴԵԼԻ ԿԱՌՈՒՑՈՒՄ ԵՎ R_t ԿՈՐԵԼԱՑԻԱՆԵՐԻ ՀԱՇՎԱՐԿ
# -----------------------------------------------------------------------------

# Անկախ կորելացիոն մատրիցա (Unconditional Correlation Matrix - Q_bar)
z_t = std_resid.values
Q_bar = np.corrcoef(z_t.T)

# DCC(1,1) ֆունկցիա՝ Q_t և R_t մատրիցների դինամիկ հաշվարկի համար
def dcc_filter(params, z, Q_bar):
    a, b = params
    T, N = z.shape
    Q_t = np.zeros((T, N, N))
    R_t = np.zeros((T, N, N))
    
    # Սկզբնական կետում Q_t-ն հավասար է Q_bar-ին
    Q_t[0] = Q_bar
    
    # Անկյունագծային տարրերի արմատները R_t ստանալու համար
    inv_sqrt_Q0 = np.diag(1 / np.sqrt(np.diag(Q_t[0])))
    R_t[0] = inv_sqrt_Q0 @ Q_t[0] @ inv_sqrt_Q0
    
    for t in range(1, T):
        # Q_t-ի դինամիկ հաշվարկ (DCC հիմնական բանաձև)
        z_prev = z[t-1].reshape(-1, 1)
        Q_t[t] = (1 - a - b) * Q_bar + a * (z_prev @ z_prev.T) + b * Q_t[t-1]
        
        # Q_t-ի ստանդարտացում R_t (կորելացիոն մատրիցա) ստանալու համար
        inv_sqrt_Qt = np.diag(1 / np.sqrt(np.diag(Q_t[t])))
        R_t[t] = inv_sqrt_Qt @ Q_t[t] @ inv_sqrt_Qt
        
    return Q_t, R_t


dcc_a = 0.04 
dcc_b = 0.94

print("Հաշվարկվում են դինամիկ պայմանական կորելացիաները (DCC)...")
Q_dynamic, R_dynamic = dcc_filter([dcc_a, dcc_b], z_t, Q_bar)

# -----------------------------------------------------------------------------
# ՔԱՅԼ 3. ԱՐԴՅՈՒՆՔՆԵՐԻ ՎԻԶՈՒԱԼԻԶԱՑԻԱ (ՄԻՋԻՆԱՑՎԱԾ ԴԻՆԱՄԻԿ ԿՈՐԵԼԱՑԻԱՆԵՐ)
# -----------------------------------------------------------------------------

# Որպեսզի ստանանք վերջնական Heatmap, հաշվում ենք ամբողջ ժամանակահատվածի միջին դինամիկ կորելացիաները
mean_dynamic_corr = np.mean(R_dynamic, axis=0)
df_dcc_mean = pd.DataFrame(mean_dynamic_corr, index=assets, columns=assets)

plt.figure(figsize=(10, 8))
sns.heatmap(df_dcc_mean, annot=True, cmap='coolwarm', vmin=-0.2, vmax=1, 
            fmt=".3f", linewidths=0.5)
plt.title('DCC-GARCH(1,1) Միջինացված Դինամիկ Կորելացիաների Մատրիցա')
plt.savefig('DCC_Heatmap_Final.png', dpi=300, bbox_inches='tight')
plt.show()
# -----------------------------------------------------------------------------
# ՔԱՅԼ 4. 3x4 ԳՐԱՖԻԿՆԵՐԻ ՑԱՆՑԸ 
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

# Առանձնացնում ենք ակտիվների խմբերը
crypto_assets = ['BTC', 'ETH', 'QNT']
trad_assets = ['S&P500', 'GOLD', 'DXY', 'US10Y']

# Տողերի գույները (Կապույտ՝ BTC, Նարնջագույն՝ ETH, Կանաչ՝ QNT)
colors = ['tab:blue', 'tab:orange', 'tab:green']

# Ստեղծում ենք 3x4 չափով նկար
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))

for i, crypto in enumerate(crypto_assets):
    crypto_idx = assets.index(crypto)
    for j, trad in enumerate(trad_assets):
        trad_idx = assets.index(trad)
        
        # Հանում ենք կոնկրետ զույգի դինամիկ կորելացիան ամբողջ ժամանակահատվածի համար
        corr_series = [R_dynamic[t, crypto_idx, trad_idx] for t in range(len(data))]
        
        ax = axes[i, j]
        ax.plot(data.index, corr_series, color=colors[i], linewidth=1.2)
        
        # Ձևավորում (վերնագրեր, սահմաններ, զրոյական գիծ)
        ax.set_title(f"{crypto} & {trad}", fontweight='bold', fontsize=12)
        ax.axhline(0, color='black', linewidth=1) # Հորիզոնական 0 գիծը
        ax.set_ylim(-1.0, 1.0) # Y առանցքի խիստ սահմանափակում
        ax.grid(True, linestyle='--', alpha=0.4)
        
        # Ուղղահայաց գծեր՝
        # 1. 2020-03-01 (COVID-19 ֆինանսական շոկ)
        ax.axvline(pd.to_datetime('2020-03-01'), color='red', linestyle='--', alpha=0.6)
        # 2. 2024-01-10 (Bitcoin ETF հաստատում)
        ax.axvline(pd.to_datetime('2024-01-10'), color='red', linestyle='--', alpha=0.6)

# Հեռավորությունների օպտիմիզացիա և ցուցադրում
plt.tight_layout()
plt.savefig('DCC_Dynamic_Correlations_3x4.png', dpi=300, bbox_inches='tight')
plt.show()


#12


import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# -------------------------------------------------------------------------
# 1. ՏՎՅԱԼՆԵՐԻ ԲԵՌՆՈՒՄ (2018 - 2026)
# -------------------------------------------------------------------------
print("Բեռնվում են տվյալները...")
tickers = {
    'BTC': 'BTC-USD',
    'DXY': 'UUP',
    'US10Y': '^TNX'
}

df_list = []
for name, ticker in tickers.items():
    temp_df = yf.download(ticker, start='2018-01-01', end='2026-04-21', progress=False)
    if isinstance(temp_df.columns, pd.MultiIndex):
        close_series = temp_df['Close'].iloc[:, 0]
    else:
        close_series = temp_df['Close']
    close_series.name = name
    df_list.append(close_series)

data = pd.concat(df_list, axis=1).ffill().dropna()
returns = data.pct_change().dropna()

# -------------------------------------------------------------------------
# 2. ԹԻՐԱԽ ԵՎ ՀԱՏԿԱՆԻՇՆԵՐ (TARGET & FEATURES)
# -------------------------------------------------------------------------
returns['Target'] = (returns['DXY'] < 0).astype(int)

features = ['BTC', 'DXY', 'US10Y']
feature_cols = []
for col in features:
    for lag in [1, 2, 3]:
        returns[f'{col}_lag{lag}'] = returns[col].shift(lag)
        feature_cols.append(f'{col}_lag{lag}')

ml_data = returns.dropna()

# -------------------------------------------------------------------------
# 3. 85/15 ԲԱԺԱՆՈՒՄ (TRAIN/TEST SPLIT)
# -------------------------------------------------------------------------

split_idx = int(len(ml_data) * 0.85)

train = ml_data.iloc[:split_idx]
test = ml_data.iloc[split_idx:]

print(f"Ընդհանուր տողերի քանակը: {len(ml_data)}")
print(f"Ուսուցման շրջան (85%): {len(train)} տող (մինչև {train.index.max().date()})")
print(f"Թեստավորման շրջան (15%): {len(test)} տող ({test.index.min().date()} - {test.index.max().date()})")

X_train, y_train = train[feature_cols], train['Target']
X_test, y_test = test[feature_cols], test['Target']

# -------------------------------------------------------------------------
# 4. XGBOOST ՄՈԴԵԼԱՎՈՐՈՒՄ
# -------------------------------------------------------------------------
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------------------------------------
# 5. ՍԻՄՈՒԼՅԱՑԻԱ ԵՎ ՀԵՋԱՎՈՐՈՒՄ
# -------------------------------------------------------------------------
test['Signal'] = model.predict(X_test)
test['Strategy_Return'] = np.where(test['Signal'] == 1, test['BTC'], test['DXY'])

test['Cum_DXY'] = (1 + test['DXY']).cumprod() - 1
test['Cum_Strategy'] = (1 + test['Strategy_Return']).cumprod() - 1

# -------------------------------------------------------------------------
# 6. ՎԻԶՈՒԱԼԻԶԱՑԻԱ
# -------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(test.index, test['Cum_DXY'] * 100, label='Պասիվ Պորտֆել (DXY)', color='salmon', lw=1.5)
plt.plot(test.index, test['Cum_Strategy'] * 100, label='XGBoost Հեջավորում (DXY $\\rightarrow$ BTC)', color='green', lw=2.5)

plt.ylabel('Կուտակային Եկամտաբերություն (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('XGBoost_Hedge_Final_Perfected.png', dpi=300)
plt.show()


#13


import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier


sns.set_style("whitegrid")
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

# Տվյալների բեռնում (2018-2026)
tickers = {'BTC': 'BTC-USD', 'SPY': 'SPY', 'DXY': 'UUP', 'GOLD': 'GLD'}
df = yf.download(list(tickers.values()), start='2018-01-01', end='2026-04-21', progress=False)['Close']
df = df.rename(columns={v: k for k, v in tickers.items()}).ffill().dropna()

# Օրական եկամտաբերություններ
returns = df.pct_change().dropna()

# Ստեղծում ենք Թիրախը (Target) XGBoost-ի համար
# Մոդելը սովորում է կանխատեսել այն օրերը, երբ BTC-ն կունենա բացասական եկամտաբերություն
returns['Target'] = (returns['BTC'] < -0.01).astype(int).shift(-1)
returns = returns.dropna()

# XGBoost Մոդելի պատրաստում
X = returns[['SPY', 'DXY', 'GOLD']]
y = returns['Target']

split = int(len(returns) * 0.85) # Վերջին 15%-ը թողնում ենք թեստավորման համար
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
model.fit(X_train, y_train)

# Ռեժիմային Փոխարկման Սիմուլյացիա
test_signals = model.predict(X_test)
test_index = returns.index[split:]

strat_returns_raw = np.where(test_signals == 1, returns['GOLD'].iloc[split:], returns['BTC'].iloc[split:])
strat_returns = pd.Series(strat_returns_raw, index=test_index)

# Կուտակային եկամտաբերություն (Capital Growth)
cum_btc = (1 + returns['BTC'].iloc[split:]).cumprod()
cum_strat = (1 + strat_returns).cumprod()

# Վերջնական Գրաֆիկ
plt.figure(figsize=(15, 8))

plt.plot(cum_btc, label='Պասիվ Ռազմավարություն (Buy & Hold BTC)', color='#9E9E9E', alpha=0.6, lw=2)
plt.plot(cum_strat, label='XGBoost Դինամիկ Հեջավորում (BTC + GOLD)', color='#1B5E20', lw=3)

# Շեշտում ենք ալգորիթմի արդյունավետությունը (Fill between)
plt.fill_between(cum_strat.index, cum_strat, cum_btc, 
                 where=(cum_strat > cum_btc), color='#C8E6C9', alpha=0.5, 
                 label='Ալգորիթմի Ավելցուկային Արժեք')

# Գրաֆիկի ձևավորում
plt.ylabel('Կապիտալի Հարաբերական Աճ', fontsize=12)
plt.legend(loc='upper left', fontsize=11, frameon=True, facecolor='white')
plt.grid(True, linestyle='--', alpha=0.3)

# Ավելացնում ենք տեղեկատվական տեքստ
plt.text(cum_strat.index[-1], cum_strat.iloc[-1], f' Final: {cum_strat.iloc[-1]:.2f}', 
         color='#1B5E20', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('XGBoost_Strategy_Final.png', dpi=300)
plt.show()
