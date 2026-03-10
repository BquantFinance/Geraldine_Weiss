import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from io import StringIO
import warnings
import pytz
import json

warnings.filterwarnings('ignore', category=FutureWarning)

st.set_page_config(
    page_title="Geraldine Weiss | Dividend Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# PREMIUM CSS - Luxury Fintech Aesthetic
# ============================================================

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg-deep: #060910;
        --bg-base: #0a0f1a;
        --bg-surface: #0f1521;
        --bg-card: #141c2f;
        --bg-card-hover: #1a2340;
        --bg-elevated: #1e2a45;
        --accent-emerald: #00e68a;
        --accent-emerald-dim: rgba(0, 230, 138, 0.15);
        --accent-emerald-glow: rgba(0, 230, 138, 0.4);
        --accent-cyan: #00b4d8;
        --accent-cyan-dim: rgba(0, 180, 216, 0.15);
        --accent-gold: #f0b429;
        --accent-gold-dim: rgba(240, 180, 41, 0.15);
        --accent-red: #ef4444;
        --accent-red-dim: rgba(239, 68, 68, 0.15);
        --text-primary: #e8edf5;
        --text-secondary: #94a3b8;
        --text-muted: #4a5568;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --border-light: rgba(255, 255, 255, 0.1);
        --glass-bg: rgba(14, 21, 33, 0.7);
        --glass-border: rgba(255, 255, 255, 0.08);
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;
    }

    /* === BASE === */
    .main { background-color: var(--bg-base) !important; }
    .block-container { padding-top: 2rem !important; max-width: 1400px !important; }
    
    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, sans-serif !important;
        color: var(--text-primary);
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }

    /* === SCROLLBAR === */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--bg-elevated); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--accent-emerald); }

    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1220 0%, #0a0f1a 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] { font-size: 14px; }

    /* === TABS === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--bg-surface);
        padding: 6px;
        border-radius: var(--radius-md);
        border: 1px solid var(--border-subtle);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        padding: 10px 20px;
        border-radius: var(--radius-sm);
        font-family: 'Outfit', sans-serif;
        font-weight: 500;
        font-size: 14px;
        letter-spacing: 0.01em;
        color: var(--text-secondary);
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg-card) !important;
        color: var(--accent-emerald) !important;
        border-bottom: none !important;
        box-shadow: 0 2px 8px rgba(0, 230, 138, 0.1);
    }

    /* === METRICS === */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 24px !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 11px !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted) !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
    }
    /* Card container for metric widget */
    [data-testid="stMetric"],
    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        padding: 14px 16px !important;
        transition: all 0.2s ease;
    }
    [data-testid="stMetric"]:hover,
    [data-testid="metric-container"]:hover {
        border-color: var(--border-light) !important;
        background: var(--bg-card-hover) !important;
    }

    /* === BUTTONS === */
    .stButton > button {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600;
        letter-spacing: 0.02em;
        border-radius: var(--radius-sm) !important;
        transition: all 0.25s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00e68a 0%, #00b4d8 100%) !important;
        color: #060910 !important;
        border: none !important;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 20px rgba(0, 230, 138, 0.3) !important;
        transform: translateY(-1px);
    }

    /* === INPUTS === */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-emerald) !important;
        box-shadow: 0 0 0 2px var(--accent-emerald-dim) !important;
    }

    /* === DATAFRAMES === */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-md);
        overflow: hidden;
    }

    /* === EXPANDER === */
    .streamlit-expanderHeader {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500;
        color: var(--text-secondary);
        background: var(--bg-surface);
        border-radius: var(--radius-sm);
    }

    /* === DIVIDER === */
    hr { border-color: var(--border-subtle) !important; opacity: 0.5; }

    /* ========================================== */
    /* CUSTOM COMPONENTS                          */
    /* ========================================== */
    
    .gw-hero {
        position: relative;
        padding: 2rem 0 1rem;
        margin-bottom: 1rem;
    }
    .gw-hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #e8edf5 0%, #00e68a 50%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        line-height: 1.2;
    }
    .gw-hero-sub {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin-top: 6px;
        letter-spacing: 0.01em;
    }
    .gw-hero-line {
        height: 2px;
        background: linear-gradient(90deg, var(--accent-emerald) 0%, var(--accent-cyan) 40%, transparent 100%);
        margin-top: 1.2rem;
        border-radius: 1px;
    }

    /* Signal Card */
    .signal-card {
        position: relative;
        text-align: center;
        padding: 36px 24px 28px;
        border-radius: var(--radius-xl);
        margin: 24px 0;
        overflow: hidden;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
    }
    .signal-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: var(--radius-xl) var(--radius-xl) 0 0;
    }
    .signal-card.buy::before { background: linear-gradient(90deg, #00e68a, #00b4d8); }
    .signal-card.sell::before { background: linear-gradient(90deg, #ef4444, #f97316); }
    .signal-card.hold::before { background: linear-gradient(90deg, #f0b429, #f59e0b); }
    .signal-card .signal-label {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.15em;
        color: var(--text-muted);
        margin-bottom: 8px;
    }
    .signal-card .signal-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }
    .signal-card .signal-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        color: var(--text-secondary);
        margin-top: 12px;
    }
    .signal-card .signal-glow {
        position: absolute;
        top: -60%; left: 50%; transform: translateX(-50%);
        width: 300px; height: 300px;
        border-radius: 50%;
        filter: blur(80px);
        opacity: 0.12;
        pointer-events: none;
    }
    .signal-card.buy .signal-glow { background: #00e68a; }
    .signal-card.sell .signal-glow { background: #ef4444; }
    .signal-card.hold .signal-glow { background: #f0b429; }

    /* Metric cards now use native st.metric with CSS above */

    /* Quality Gauge */
    .quality-display {
        display: flex;
        align-items: center;
        gap: 20px;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 20px 24px;
        margin: 16px 0;
    }
    .quality-grade {
        width: 72px; height: 72px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 900;
        flex-shrink: 0;
    }
    .quality-bars {
        flex: 1;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
    }
    .quality-bar-item {
        text-align: center;
    }
    .quality-bar-item .qb-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
        margin-bottom: 6px;
        font-family: 'Outfit', sans-serif;
    }
    .quality-bar-track {
        height: 6px;
        background: var(--bg-deep);
        border-radius: 3px;
        overflow: hidden;
        margin-bottom: 4px;
    }
    .quality-bar-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.6s ease;
    }
    .quality-bar-item .qb-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: var(--text-secondary);
    }

    /* Badges */
    .badge-row { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin: 8px 0; }
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 4px 10px;
        border-radius: 20px;
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.03em;
        border: 1px solid;
    }
    .badge-source-dh { background: var(--accent-emerald-dim); color: var(--accent-emerald); border-color: rgba(0,230,138,0.2); }
    .badge-source-yf { background: var(--accent-cyan-dim); color: var(--accent-cyan); border-color: rgba(0,180,216,0.2); }
    .badge-conf-high { background: var(--accent-emerald-dim); color: var(--accent-emerald); border-color: rgba(0,230,138,0.2); }
    .badge-conf-medium { background: var(--accent-gold-dim); color: var(--accent-gold); border-color: rgba(240,180,41,0.2); }
    .badge-conf-low { background: var(--accent-red-dim); color: var(--accent-red); border-color: rgba(239,68,68,0.2); }

    /* Info boxes */
    .insight-box {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid;
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 14px;
        color: var(--text-secondary);
        line-height: 1.6;
    }
    .insight-box.green { border-left-color: var(--accent-emerald); }
    .insight-box.red { border-left-color: var(--accent-red); }
    .insight-box.gold { border-left-color: var(--accent-gold); }
    .insight-box strong { color: var(--text-primary); }

    /* Projection card */
    .projection-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 10px;
        margin: 12px 0;
    }
    .projection-item {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-sm);
        padding: 14px;
        text-align: center;
    }
    .projection-item .pi-year {
        font-family: 'Outfit', sans-serif;
        font-size: 11px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .projection-item .pi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-emerald);
        margin-top: 4px;
    }
    .projection-item .pi-yield {
        font-size: 11px;
        color: var(--text-secondary);
        margin-top: 2px;
    }

    /* Footer */
    .gw-footer {
        position: fixed;
        bottom: 16px;
        right: 16px;
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        padding: 8px 16px;
        border-radius: var(--radius-sm);
        border: 1px solid var(--glass-border);
        z-index: 999;
        font-family: 'DM Sans', sans-serif;
        font-size: 12px;
        color: var(--text-secondary);
    }
    .gw-footer a { color: var(--accent-emerald); text-decoration: none; font-weight: 600; }
    .gw-footer a:hover { text-decoration: underline; }

    /* Disclaimer */
    .disclaimer {
        font-size: 11px;
        color: var(--text-muted);
        font-style: italic;
        padding: 12px 16px;
        background: var(--bg-surface);
        border-radius: var(--radius-sm);
        border: 1px solid var(--border-subtle);
        margin-top: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# DATA FETCHER
# ============================================================

class DividendDataFetcher:
    def __init__(self):
        self.base_url = "https://dividendhistory.org/payout"
        self.session = requests.Session()
        self.cache = {}

    def fetch_dividends(self, ticker, start_date=None, end_date=None):
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        url = f"{self.base_url}/{ticker}/"
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                       'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'}
            response = self.session.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()
            if len(response.text) < 100: return pd.DataFrame()
            try: tables = pd.read_html(StringIO(response.text))
            except: return pd.DataFrame()
            if not tables: return pd.DataFrame()
            df = None
            for table in tables:
                temp = table.copy()
                temp.columns = [str(c).strip() for c in temp.columns]
                if all(c.isdigit() for c in temp.columns):
                    temp.columns = temp.iloc[0].astype(str).str.strip().tolist()
                    temp = temp.iloc[1:].reset_index(drop=True)
                if 'Ex-Dividend Date' in temp.columns or 'Cash Amount' in temp.columns:
                    df = temp; break
            if df is None: return pd.DataFrame()
            df = df.rename(columns={'Ex-Dividend Date':'ex_dividend_date','Payout Date':'payout_date',
                                    'Cash Amount':'amount','% Change':'pct_change'})
            if 'ex_dividend_date' not in df.columns: return pd.DataFrame()
            df['ex_dividend_date'] = pd.to_datetime(df['ex_dividend_date'], errors='coerce')
            if 'payout_date' in df.columns: df['payout_date'] = pd.to_datetime(df['payout_date'], errors='coerce')
            if 'amount' in df.columns:
                df['amount'] = df['amount'].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).str.strip()
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            df = df[df['ex_dividend_date'].notna() & df['amount'].notna() & (df['amount']>0)]
            if start_date: df = df[df['ex_dividend_date'] >= pd.to_datetime(start_date)]
            if end_date: df = df[df['ex_dividend_date'] <= pd.to_datetime(end_date)]
            df = df.sort_values('ex_dividend_date', ascending=False).reset_index(drop=True)
            if not df.empty: self.cache[cache_key] = df.copy()
            return df
        except: return pd.DataFrame()


# ============================================================
# ANALYZER
# ============================================================

class GeraldineWeissAnalyzer:
    def __init__(self, ticker, years=6):
        self.ticker = ticker
        self.years = years
        self.dividend_fetcher = DividendDataFetcher()
        self.data_source = None

    def fetch_price_data(self):
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        try:
            t = yf.Ticker(self.ticker)
            data = t.history(start=start_date, end=end_date, auto_adjust=False, actions=False)
            if data.empty: return None
            if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(0)
            data.index = pd.to_datetime(data.index)
            if isinstance(data.index, pd.DatetimeIndex) and data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            return data
        except: return None

    def fetch_dividend_data(self):
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        if '.' not in self.ticker:
            try:
                df = self.dividend_fetcher.fetch_dividends(self.ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
                if not df.empty: self.data_source = "dividendhistory.org"; return df
            except: pass
        df = self._fetch_from_yfinance(start_date)
        if not df.empty: self.data_source = "yfinance"; return df
        self.data_source = "none"; return pd.DataFrame()

    def _fetch_from_yfinance(self, start_date):
        try:
            t = yf.Ticker(self.ticker)
            divs = t.dividends
            if divs.empty:
                try:
                    actions = t.actions
                    if 'Dividends' in actions.columns: divs = actions['Dividends'][actions['Dividends']>0]
                except: pass
            if not divs.empty:
                if divs.index.tz is not None:
                    if start_date.tzinfo is None: start_date = pytz.UTC.localize(start_date)
                    start_date = start_date.astimezone(divs.index.tz)
                else:
                    if start_date.tzinfo is not None: start_date = start_date.replace(tzinfo=None)
                divs = divs[divs.index >= start_date]
                if not divs.empty:
                    df = pd.DataFrame({'ex_dividend_date': divs.index.tz_localize(None) if divs.index.tz else divs.index, 'amount': divs.values})
                    return df[df['amount']>0].sort_values('ex_dividend_date', ascending=False).reset_index(drop=True)
            return pd.DataFrame()
        except: return pd.DataFrame()

    def calculate_ttm_dividends(self, prices, dividend_df):
        if dividend_df.empty or prices is None or prices.empty: return None
        prices = prices.copy()
        if isinstance(prices.index, pd.DatetimeIndex) and prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)
        if 'Close' not in prices.columns:
            if 'close' in prices.columns: prices = prices.rename(columns={'close':'Close'})
            else: return None
        div_s = dividend_df.copy().sort_values('ex_dividend_date').set_index('ex_dividend_date')['amount']
        div_s.index = pd.to_datetime(div_s.index)
        if div_s.index.tz is not None: div_s.index = div_s.index.tz_localize(None)
        m, s = div_s.mean(), div_s.std()
        if s > 0: div_s = div_s[abs(div_s - m) <= 2.5 * s]
        ttm = []
        dd, da = div_s.index, div_s.values
        for d in prices.index:
            lb = d - timedelta(days=365)
            mask = (dd > lb) & (dd <= d)
            ttm.append(da[mask].sum())
        prices['ttm_dividend'] = ttm
        prices = prices[prices['ttm_dividend'] > 0]
        return prices if not prices.empty else None

    def calculate_valuation_bands(self, ptm):
        if ptm is None or ptm.empty: return None
        m = ptm.copy()
        m['div_yield'] = m['ttm_dividend'] / m['Close']
        ym, ys = m['div_yield'].median(), m['div_yield'].std()
        if ys > 0: m = m[abs(m['div_yield'] - ym) <= 3 * ys]
        if m.empty or len(m) < 10: return None
        mx, mn = m['div_yield'].quantile(0.95), m['div_yield'].quantile(0.05)
        if mx <= 0 or mn <= 0 or mx <= mn: return None
        m['undervalued_raw'] = (m['div_yield'] / mx) * m['Close']
        m['overvalued_raw'] = (m['div_yield'] / mn) * m['Close']
        # Suavizar bandas con rolling median para eliminar picos por saltos discretos del TTM
        window = min(21, max(5, len(m) // 50))  # ventana adaptativa
        m['undervalued'] = m['undervalued_raw'].rolling(window=window, center=True, min_periods=1).median()
        m['overvalued'] = m['overvalued_raw'].rolling(window=window, center=True, min_periods=1).median()
        # Segunda pasada EWM para suavizado extra
        m['undervalued'] = m['undervalued'].ewm(span=window, adjust=False).mean()
        m['overvalued'] = m['overvalued'].ewm(span=window, adjust=False).mean()
        m.attrs['max_yield'] = mx; m.attrs['min_yield'] = mn; m.attrs['median_yield'] = ym
        return m

    def get_current_signal(self, adf):
        if adf is None or adf.empty: return "DESCONOCIDO", "", 0
        l = adf.iloc[-1]
        p, uv, ov = l['Close'], l['undervalued'], l['overvalued']
        rs = ov - uv
        if rs <= 0: return "DESCONOCIDO", "", 0
        bz, sz = uv + rs*0.2, ov - rs*0.2
        sc = ((ov - p) / rs) * 200 - 100
        if p <= bz: return "COMPRA FUERTE", f"En zona infravalorada", sc
        elif p <= uv: return "COMPRA", f"Cerca de infravalorada", sc
        elif p >= sz: return "VENTA FUERTE", f"En zona sobrevalorada", sc
        elif p >= ov: return "VENTA", f"Cerca de sobrevalorada", sc
        else: return "MANTENER", f"En valor razonable", sc

    def calculate_quality_score(self, ddf):
        empty = {'total_score':0,'grade':'F','details':{'consecutive_years':0,'n_years':0,'years_score':0,'growth_pct':0,'growth_score':0,'cv':0,'stability_score':0,'payments_per_year':0,'frequency_score':0}}
        if ddf.empty: return empty
        div = ddf.copy().sort_values('ex_dividend_date', ascending=True)
        div['year'] = div['ex_dividend_date'].dt.year
        uy = sorted(div['year'].unique()); ny = len(uy)
        cons, mx = 1, 1
        for i in range(1, len(uy)):
            if uy[i] == uy[i-1]+1: cons += 1; mx = max(mx, cons)
            else: cons = 1
        ys = min(30, mx * 5)
        ann = div.groupby('year')['amount'].sum().sort_index()
        if len(ann) >= 2:
            gy = sum(1 for i in range(1, len(ann)) if ann.iloc[i] >= ann.iloc[i-1])
            gp = gy / (len(ann)-1); gs = int(gp * 30)
        else: gp, gs = 0, 0
        if len(ann) >= 2 and ann.mean() > 0:
            cv = ann.std() / ann.mean()
            ss = 20 if cv<0.1 else 15 if cv<0.2 else 10 if cv<0.35 else 5 if cv<0.5 else 0
        else: cv, ss = 0, 0
        ppy = len(div) / max(ny, 1)
        fs = 20 if ppy>=3.5 else 15 if ppy>=1.8 else 10 if ppy>=0.9 else 5
        tot = ys+gs+ss+fs
        gr = 'A' if tot>=80 else 'B' if tot>=60 else 'C' if tot>=40 else 'D' if tot>=20 else 'F'
        return {'total_score':tot,'grade':gr,'details':{'consecutive_years':mx,'n_years':ny,'years_score':ys,'growth_pct':gp*100,'growth_score':gs,'cv':cv,'stability_score':ss,'payments_per_year':ppy,'frequency_score':fs}}

    def calculate_confidence(self, ddf, adf):
        if ddf.empty or adf is None or adf.empty: return 'low','Baja'
        np_, ny, nd = len(ddf), ddf['ex_dividend_date'].dt.year.nunique(), len(adf)
        sc = (3 if np_>=20 else 2 if np_>=12 else 1 if np_>=6 else 0)
        sc += (3 if ny>=5 else 2 if ny>=3 else 1 if ny>=2 else 0)
        sc += (2 if nd>=1000 else 1 if nd>=500 else 0)
        if sc>=7: return 'high','Alta'
        elif sc>=4: return 'medium','Media'
        return 'low','Baja'

    def backtest_signals(self, adf):
        if adf is None or adf.empty or len(adf)<50: return None
        df = adf.sort_index(); trades = []; pos = None
        for i in range(len(df)):
            r = df.iloc[i]; d = df.index[i]
            p, uv, ov = r['Close'], r['undervalued'], r['overvalued']
            rs = ov - uv
            if rs <= 0: continue
            bz, sz = uv + rs*0.2, ov - rs*0.2
            if pos is None and p <= bz: pos = {'ed':d,'ep':p}
            elif pos is not None and p >= sz:
                ret = (p/pos['ep']-1)*100; hd = (d-pos['ed']).days
                trades.append({'entry_date':pos['ed'],'entry_price':pos['ep'],'exit_date':d,'exit_price':p,'return_pct':ret,'holding_days':hd,'open':False}); pos = None
        if pos:
            lr = df.iloc[-1]; ret = (lr['Close']/pos['ep']-1)*100; hd = (df.index[-1]-pos['ed']).days
            trades.append({'entry_date':pos['ed'],'entry_price':pos['ep'],'exit_date':df.index[-1],'exit_price':lr['Close'],'return_pct':ret,'holding_days':hd,'open':True})
        if not trades: return None
        tdf = pd.DataFrame(trades); ct = tdf[~tdf['open'].astype(bool)]
        s = {'total_trades':len(tdf),'closed_trades':len(ct),'open_trades':len(tdf)-len(ct)}
        if not ct.empty:
            s['win_rate'] = (ct['return_pct']>0).mean()*100; s['avg_return'] = ct['return_pct'].mean()
            s['median_return'] = ct['return_pct'].median(); s['max_return'] = ct['return_pct'].max()
            s['min_return'] = ct['return_pct'].min(); s['avg_holding_days'] = ct['holding_days'].mean()
            cum = 1.0
            for r in ct['return_pct']: cum *= (1+r/100)
            s['cumulative_return'] = (cum-1)*100
        else:
            for k in ['win_rate','avg_return','median_return','max_return','min_return','avg_holding_days','cumulative_return']: s[k]=0
        return {'trades':tdf,'stats':s}

    def project_dividend_income(self, current_ttm, cagr, price, investment=10000, years_ahead=5):
        """MEJORA: Proyección de ingresos por dividendos"""
        if current_ttm <= 0 or price <= 0: return None
        shares = investment / price
        projections = []
        for y in range(years_ahead + 1):
            projected_div = current_ttm * ((1 + cagr/100) ** y) if cagr > -50 else current_ttm
            annual_income = shares * projected_div
            yoc = (projected_div / price) * 100
            projections.append({'year': y, 'annual_div': projected_div, 'income': annual_income, 'yoc': yoc})
        return projections


# ============================================================
# CACHED ANALYSIS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_ticker_quick(ticker, years=6):
    try:
        a = GeraldineWeissAnalyzer(ticker, years)
        pd_ = a.fetch_price_data()
        if pd_ is None or pd_.empty: return None
        dd = a.fetch_dividend_data()
        if dd.empty: return None
        pttm = a.calculate_ttm_dividends(pd_, dd)
        if pttm is None: return None
        adf = a.calculate_valuation_bands(pttm)
        if adf is None or adf.empty: return None
        sig, desc, sc = a.get_current_signal(adf)
        lat = adf.iloc[-1]
        ds = dd.copy().sort_values('ex_dividend_date')
        ds['year'] = ds['ex_dividend_date'].dt.year
        ann = ds.groupby('year')['amount'].sum().sort_index()
        cagr = 0
        if len(ann) > 1:
            f, l, n = ann.iloc[0], ann.iloc[-1], len(ann)-1
            if f > 0 and n > 0: cagr = ((l/f)**(1/n)-1)*100
        q = a.calculate_quality_score(dd)
        cl, clb = a.calculate_confidence(dd, adf)
        bt = a.backtest_signals(adf)
        proj = a.project_dividend_income(lat['ttm_dividend'], cagr, lat['Close'])
        return {'ticker':ticker,'price':lat['Close'],'yield':lat['div_yield']*100,'ttm_dividend':lat['ttm_dividend'],
                'undervalued':lat['undervalued'],'overvalued':lat['overvalued'],'signal':sig,'description':desc,'score':sc,
                'cagr':cagr,'analysis_df':adf,'dividend_data':dd,'data_source':a.data_source,'quality':q,
                'confidence_level':cl,'confidence_label':clb,'backtest':bt,'projection':proj}
    except: return None


# ============================================================
# PORTFOLIO UTILS
# ============================================================

def create_weighted_portfolio_analysis(pr):
    dates = set()
    for r in pr: dates.update(r['analysis_df'].index.tolist())
    dates = sorted(dates)
    pdf = pd.DataFrame(index=pd.DatetimeIndex(dates))
    pdf['weighted_price'] = 0.0; pdf['weighted_undervalued'] = 0.0; pdf['weighted_overvalued'] = 0.0
    for d in dates:
        tw, wp, wu, wo = 0, 0, 0, 0
        for r in pr:
            df = r['analysis_df']
            if d in df.index:
                row = df.loc[d]
                if isinstance(row, pd.DataFrame): row = row.iloc[-1]
            else:
                av = df.index[df.index <= d]
                if len(av) > 0:
                    row = df.loc[av[-1]]
                    if isinstance(row, pd.DataFrame): row = row.iloc[-1]
                else: continue
            w = r['portfolio_weight'] / 100; tw += w
            wp += row['Close']*w; wu += row['undervalued']*w; wo += row['overvalued']*w
        if tw > 0:
            pdf.loc[d,'weighted_price'] = wp/tw; pdf.loc[d,'weighted_undervalued'] = wu/tw; pdf.loc[d,'weighted_overvalued'] = wo/tw
    return pdf[(pdf!=0).all(axis=1)]

def portfolio_to_json(pdf): return json.dumps(pdf.to_dict(orient='records'), indent=2, ensure_ascii=False)
def json_to_portfolio(js):
    try:
        df = pd.DataFrame(json.loads(js))
        if 'ticker' in df.columns and 'weight' in df.columns:
            df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(0)
            return df[['ticker','weight']]
    except: pass
    return None


# ============================================================
# HTML COMPONENTS
# ============================================================

def render_hero():
    st.markdown("""
    <div class="gw-hero">
        <h1 class="gw-hero-title">Geraldine Weiss</h1>
        <div class="gw-hero-sub">Dividend Intelligence Platform — Valoración profesional por rentabilidad de dividendos</div>
        <div class="gw-hero-line"></div>
    </div>""", unsafe_allow_html=True)


def render_signal(signal, price, description):
    cls = 'buy' if 'COMPRA' in signal else 'sell' if 'VENTA' in signal else 'hold'
    colors = {"COMPRA FUERTE":"var(--accent-emerald)","COMPRA":"#51cf66","MANTENER":"var(--accent-gold)","VENTA":"#f97316","VENTA FUERTE":"var(--accent-red)"}
    c = colors.get(signal, 'var(--text-primary)')
    st.markdown(f"""
    <div class="signal-card {cls}">
        <div class="signal-glow"></div>
        <div class="signal-label">Señal de valoración</div>
        <div class="signal-value" style="color:{c}">{signal}</div>
        <div class="signal-price">&#36;{price:.2f} — {description}</div>
    </div>""", unsafe_allow_html=True)


def render_metrics(items):
    """Render metrics using native st.columns + st.metric for guaranteed rendering"""
    cols = st.columns(len(items))
    for i, it in enumerate(items):
        label = it['label']
        value = str(it['value']).replace('$', '')  # st.metric handles display
        delta = it.get('delta')
        # st.metric delta_color: "normal" = green up / red down, "inverse" = opposite, "off" = grey
        delta_color = "off"
        if delta:
            dt = it.get('delta_type', 'neutral')
            if dt == 'positive':
                delta_color = "normal"
            elif dt == 'negative':
                delta_color = "inverse"
        cols[i].metric(label=label, value=value, delta=delta, delta_color=delta_color)


def render_badges(source, confidence_level, confidence_label):
    scls = 'badge-source-dh' if source == 'dividendhistory.org' else 'badge-source-yf'
    sname = 'dividendhistory.org' if source == 'dividendhistory.org' else 'yfinance'
    ccls = f'badge-conf-{confidence_level}'
    st.markdown(f"""
    <div class="badge-row">
        <span class="badge {scls}">📊 {sname}</span>
        <span class="badge {ccls}">🎯 Confianza: {confidence_label}</span>
    </div>""", unsafe_allow_html=True)


def render_quality(q):
    gr = q['grade']; tot = q['total_score']; d = q['details']
    gc = {'A':'var(--accent-emerald)','B':'#51cf66','C':'var(--accent-gold)','D':'#f97316','F':'var(--accent-red)'}.get(gr,'#fff')
    bars = [
        ('Historial', d['years_score'], 30, 'var(--accent-emerald)'),
        ('Crecimiento', d['growth_score'], 30, 'var(--accent-cyan)'),
        ('Estabilidad', d['stability_score'], 20, 'var(--accent-gold)'),
        ('Frecuencia', d['frequency_score'], 20, '#a78bfa'),
    ]
    bars_html = ""
    for label, val, mx, color in bars:
        pct = (val/mx)*100
        bars_html += f"""<div class="quality-bar-item">
            <div class="qb-label">{label}</div>
            <div class="quality-bar-track"><div class="quality-bar-fill" style="width:{pct}%;background:{color}"></div></div>
            <div class="qb-val">{val}/{mx}</div>
        </div>"""
    st.markdown(f"""
    <div class="quality-display">
        <div class="quality-grade" style="background:rgba(255,255,255,0.05);border:3px solid {gc};color:{gc}">{gr}</div>
        <div style="flex:1">
            <div style="font-family:'Outfit',sans-serif;font-size:13px;color:var(--text-secondary);margin-bottom:8px">
                Quality Score <span style="color:{gc};font-weight:700">{tot}/100</span>
                · {d['consecutive_years']} años consecutivos · {d['payments_per_year']:.1f} pagos/año · Crecimiento {d['growth_pct']:.0f}%
            </div>
            <div class="quality-bars">{bars_html}</div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_projection(proj, investment=10000):
    if not proj: return
    items = ""
    for p in proj:
        yr = "Hoy" if p['year']==0 else f"Año {p['year']}"
        items += f"""<div class="projection-item">
            <div class="pi-year">{yr}</div>
            <div class="pi-value">&#36;{p['income']:.0f}</div>
            <div class="pi-yield">YoC {p['yoc']:.2f}%</div>
        </div>"""
    st.markdown(f"""
    <div style="font-family:'Outfit',sans-serif;font-size:13px;color:var(--text-secondary);margin-bottom:8px">
        Proyección de ingresos anuales por dividendos · Inversión: &#36;{investment:,.0f}
    </div>
    <div class="projection-grid">{items}</div>""", unsafe_allow_html=True)


# ============================================================
# PLOTLY CHARTS (Premium theme)
# ============================================================

CHART_COLORS = {
    'bg': '#0a0f1a', 'grid': 'rgba(255,255,255,0.04)', 'text': '#94a3b8',
    'emerald': '#00e68a', 'cyan': '#00b4d8', 'red': '#ef4444', 'gold': '#f0b429',
    'emerald_dim': 'rgba(0,230,138,0.1)', 'red_dim': 'rgba(239,68,68,0.08)',
}

def _base_layout(title='', height=520):
    return dict(
        template='plotly_dark', plot_bgcolor=CHART_COLORS['bg'], paper_bgcolor=CHART_COLORS['bg'],
        height=height, hovermode='x unified', margin=dict(l=60, r=30, t=70, b=50),
        title=dict(text=title, font=dict(family='Outfit, sans-serif', size=18, color='#e8edf5'), x=0.5, xanchor='center'),
        xaxis=dict(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False, tickformat='%b %Y',
                   tickfont=dict(family='JetBrains Mono, monospace', size=10, color=CHART_COLORS['text'])),
        yaxis=dict(gridcolor=CHART_COLORS['grid'], showgrid=True, zeroline=False,
                   tickfont=dict(family='JetBrains Mono, monospace', size=10, color=CHART_COLORS['text'])),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family='DM Sans, sans-serif', size=11, color=CHART_COLORS['text']),
                    bgcolor='rgba(14,21,33,0.8)', bordercolor='rgba(255,255,255,0.06)', borderwidth=1),
        hoverlabel=dict(bgcolor='#141c2f', bordercolor='rgba(255,255,255,0.1)', font=dict(family='DM Sans, sans-serif', size=12)),
    )


def plot_valuation(adf, ticker):
    adf = adf.copy()
    if isinstance(adf.index, pd.DatetimeIndex) and adf.index.tz: adf.index = adf.index.tz_localize(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=adf.index, y=adf['overvalued'], line=dict(color='rgba(0,0,0,0)',width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=adf.index, y=adf['undervalued'], name='Rango Valor', fill='tonexty', fillcolor=CHART_COLORS['emerald_dim'], line=dict(color='rgba(0,0,0,0)',width=0), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=adf.index, y=adf['overvalued'], name='Sobrevalorada', line=dict(color=CHART_COLORS['red'],width=2.5,dash='dot'), hovertemplate='%{y:.2f}<extra>Sobrev.</extra>'))
    fig.add_trace(go.Scatter(x=adf.index, y=adf['undervalued'], name='Infravalorada', line=dict(color=CHART_COLORS['emerald'],width=2.5,dash='dot'), hovertemplate='%{y:.2f}<extra>Infrav.</extra>'))
    fig.add_trace(go.Scatter(x=adf.index, y=adf['Close'], name='Precio', line=dict(color=CHART_COLORS['cyan'],width=3), hovertemplate='$%{y:.2f}<extra>Precio</extra>'))
    lat = adf.iloc[-1]
    fig.add_trace(go.Scatter(x=[adf.index[-1]], y=[lat['Close']], mode='markers', marker=dict(size=10,color=CHART_COLORS['cyan'],line=dict(color='white',width=2)), showlegend=False))
    fig.update_layout(**_base_layout(f'{ticker} — Bandas de Valoración Weiss'))
    return fig


def plot_yield(adf, ticker):
    adf = adf.copy()
    if isinstance(adf.index, pd.DatetimeIndex) and adf.index.tz: adf.index = adf.index.tz_localize(None)
    yp = adf['div_yield']*100; p95,p50,p05 = yp.quantile(0.95),yp.median(),yp.quantile(0.05)
    fig = go.Figure()
    fig.add_hrect(y0=p95*0.9, y1=yp.max()*1.05, fillcolor="rgba(0,230,138,0.06)", line_width=0)
    fig.add_hrect(y0=yp.min()*0.95, y1=p05*1.1, fillcolor="rgba(239,68,68,0.05)", line_width=0)
    fig.add_hline(y=p95, line=dict(color=CHART_COLORS['emerald'],width=1.5,dash='dash'), annotation_text=f"P95 {p95:.2f}%", annotation_position="right", annotation_font=dict(color=CHART_COLORS['emerald'],size=10,family='JetBrains Mono'))
    fig.add_hline(y=p50, line=dict(color=CHART_COLORS['gold'],width=1.5,dash='dot'), annotation_text=f"Med {p50:.2f}%", annotation_position="right", annotation_font=dict(color=CHART_COLORS['gold'],size=10,family='JetBrains Mono'))
    fig.add_hline(y=p05, line=dict(color=CHART_COLORS['red'],width=1.5,dash='dash'), annotation_text=f"P5 {p05:.2f}%", annotation_position="right", annotation_font=dict(color=CHART_COLORS['red'],size=10,family='JetBrains Mono'))
    fig.add_trace(go.Scatter(x=adf.index, y=yp, name='Yield TTM', line=dict(color=CHART_COLORS['cyan'],width=2.5), fill='tozeroy', fillcolor='rgba(0,180,216,0.08)', hovertemplate='%{y:.2f}%<extra>Yield</extra>'))
    fig.add_trace(go.Scatter(x=[adf.index[-1]], y=[yp.iloc[-1]], mode='markers', marker=dict(size=10,color=CHART_COLORS['cyan'],line=dict(color='white',width=2)), showlegend=False))
    fig.update_layout(**_base_layout(f'{ticker} — Dividend Yield TTM', 420))
    fig.update_yaxes(ticksuffix='%')
    return fig


def plot_backtest_chart(adf, bt, ticker):
    if bt is None: return None
    adf = adf.copy(); trades = bt['trades']; fig = go.Figure()
    fig.add_trace(go.Scatter(x=adf.index, y=adf['Close'], name='Precio', line=dict(color='rgba(0,180,216,0.4)',width=2)))
    fig.add_trace(go.Scatter(x=adf.index, y=adf['overvalued'], name='Sobrev.', line=dict(color='rgba(239,68,68,0.2)',width=1,dash='dot'), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=adf.index, y=adf['undervalued'], name='Infrav.', line=dict(color='rgba(0,230,138,0.2)',width=1,dash='dot'), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=trades['entry_date'], y=trades['entry_price'], mode='markers', name='Compra', marker=dict(size=12,color=CHART_COLORS['emerald'],symbol='triangle-up',line=dict(color='white',width=1.5))))
    fig.add_trace(go.Scatter(x=trades['exit_date'], y=trades['exit_price'], mode='markers', name='Venta', marker=dict(size=12,color=CHART_COLORS['red'],symbol='triangle-down',line=dict(color='white',width=1.5))))
    for _, t in trades.iterrows():
        c = CHART_COLORS['emerald'] if t['return_pct']>0 else CHART_COLORS['red']
        fig.add_trace(go.Scatter(x=[t['entry_date'],t['exit_date']], y=[t['entry_price'],t['exit_price']], mode='lines', line=dict(color=c,width=1,dash='dash'), showlegend=False, hoverinfo='skip'))
    fig.update_layout(**_base_layout(f'{ticker} — Backtest', 480))
    return fig


def plot_portfolio_chart(pdf):
    pdf = pdf.copy()
    if isinstance(pdf.index, pd.DatetimeIndex) and pdf.index.tz: pdf.index = pdf.index.tz_localize(None)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['weighted_overvalued'], line=dict(color='rgba(0,0,0,0)',width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['weighted_undervalued'], name='Rango Valor', fill='tonexty', fillcolor=CHART_COLORS['emerald_dim'], line=dict(color='rgba(0,0,0,0)',width=0), hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['weighted_overvalued'], name='Sobrevalorada', line=dict(color=CHART_COLORS['red'],width=2.5,dash='dot')))
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['weighted_undervalued'], name='Infravalorada', line=dict(color=CHART_COLORS['emerald'],width=2.5,dash='dot')))
    fig.add_trace(go.Scatter(x=pdf.index, y=pdf['weighted_price'], name='Ponderado', line=dict(color=CHART_COLORS['cyan'],width=3)))
    lat = pdf.iloc[-1]
    fig.add_trace(go.Scatter(x=[pdf.index[-1]], y=[lat['weighted_price']], mode='markers', marker=dict(size=10,color=CHART_COLORS['cyan'],line=dict(color='white',width=2)), showlegend=False))
    fig.update_layout(**_base_layout('Cartera Ponderada — Valoración Weiss'))
    return fig


def plot_comparison(results):
    fig = go.Figure()
    tks = [r['ticker'] for r in results]; ps = [r['price'] for r in results]; uvs = [r['undervalued'] for r in results]; ovs = [r['overvalued'] for r in results]
    x = list(range(len(tks)))
    fig.add_trace(go.Bar(x=x, y=[ov-uv for ov,uv in zip(ovs,uvs)], base=uvs, name='Rango Valor', marker_color='rgba(0,230,138,0.1)', marker_line=dict(color='rgba(0,230,138,0.3)',width=1), hoverinfo='skip'))
    cols = [CHART_COLORS['emerald'] if 'COMPRA' in r['signal'] else CHART_COLORS['red'] if 'VENTA' in r['signal'] else CHART_COLORS['gold'] for r in results]
    fig.add_trace(go.Scatter(x=x, y=ps, name='Precio', mode='markers+text', marker=dict(size=16,color=cols,line=dict(color='white',width=2),symbol='diamond'), text=[f"${p:.1f}" for p in ps], textposition='top center', textfont=dict(family='JetBrains Mono',size=11,color='#e8edf5')))
    fig.add_trace(go.Scatter(x=x, y=ovs, name='Sobrev.', mode='markers', marker=dict(size=8,color=CHART_COLORS['red'],symbol='line-ew-open',line=dict(width=2,color=CHART_COLORS['red']))))
    fig.add_trace(go.Scatter(x=x, y=uvs, name='Infrav.', mode='markers', marker=dict(size=8,color=CHART_COLORS['emerald'],symbol='line-ew-open',line=dict(width=2,color=CHART_COLORS['emerald']))))
    fig.update_layout(**_base_layout('Comparación de Valoración', 460))
    fig.update_xaxes(tickmode='array', tickvals=x, ticktext=tks, tickformat=None)
    return fig


def plot_pie(pdata):
    fig = go.Figure()
    colors = ['#00e68a','#00b4d8','#7b2ff7','#f0b429','#ef4444','#51cf66','#f97316','#06b6d4','#a78bfa','#fb7185']
    fig.add_trace(go.Pie(labels=pdata['ticker'], values=pdata['weight'],
        marker=dict(colors=colors[:len(pdata)], line=dict(color=CHART_COLORS['bg'],width=3)),
        textinfo='label+percent', textfont=dict(family='Outfit, sans-serif',size=13,color='white'),
        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<extra></extra>', hole=0.45))
    fig.update_layout(template='plotly_dark', height=400, plot_bgcolor=CHART_COLORS['bg'], paper_bgcolor=CHART_COLORS['bg'],
                      showlegend=False, margin=dict(l=20,r=20,t=20,b=20),
                      annotations=[dict(text='Cartera', font=dict(family='Outfit',size=16,color=CHART_COLORS['text']), showarrow=False)])
    return fig


def plot_dividend_bars(dd):
    da = dd.copy(); da['year'] = dd['ex_dividend_date'].dt.year
    da = da.groupby('year')['amount'].sum().sort_index()
    colors = [CHART_COLORS['emerald'] if i==len(da)-1 else 'rgba(0,230,138,0.4)' for i in range(len(da))]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=da.index, y=da.values, marker_color=colors, marker_line=dict(color=CHART_COLORS['emerald'],width=1),
                         hovertemplate='<b>%{x}</b><br>$%{y:.4f}<extra></extra>'))
    fig.update_layout(**_base_layout('Dividendo Anual', 320))
    fig.update_xaxes(dtick=1, tickformat='d')
    fig.update_yaxes(title='Dividendo ($)')
    return fig


# ============================================================
# MAIN APP
# ============================================================

def main():
    render_hero()

    tab1, tab2, tab3 = st.tabs(["🎯  Análisis Individual", "📊  Comparación", "💼  Cartera"])

    # ==================== TAB 1 ====================
    with tab1:
        cs, cm = st.columns([1, 3])
        with cs:
            st.markdown("#### ⚙️ Configuración")
            ticker = st.text_input("Ticker", value="KO", help="Símbolo de acción")
            years = st.slider("Período (años)", 3, 10, 6)
            st.divider()
            analyze_btn = st.button("🔍  Analizar", type="primary", use_container_width=True)
            st.divider()
            with st.expander("💡 Método Weiss"):
                st.markdown("""
                Identifica valor mediante dividend yield histórico:
                **Alto yield** → Infravalorada · **Bajo yield** → Sobrevalorada
                
                Usa **TTM** (trailing 12 meses) para precisión real.
                """)
            st.divider()
            st.caption("**Sugeridos**")
            st.caption("🇺🇸 KO · JNJ · PG · MMM · XOM")
            st.caption("🇪🇸 IBE.MC · SAN.MC · TEF.MC")
            st.caption("🇬🇧 BP.L · ULVR.L · HSBA.L")
            st.caption("🇨🇦 RY.TO · TD.TO · ENB.TO")

        with cm:
            if analyze_btn and ticker:
                with st.spinner('Analizando...'):
                    r = analyze_ticker_quick(ticker.upper(), years)
                if r is None:
                    st.error(f"❌ Sin datos para **{ticker.upper()}**. Verifica ticker, período o dividendos.")
                else:
                    render_badges(r['data_source'], r['confidence_level'], r['confidence_label'])
                    render_signal(r['signal'], r['price'], r['description'])

                    render_metrics([
                        {'label':'Precio','value':f"${r['price']:.2f}"},
                        {'label':'Yield TTM','value':f"{r['yield']:.2f}%"},
                        {'label':'Dividendo TTM','value':f"${r['ttm_dividend']:.2f}"},
                        {'label':'Infravalorada','value':f"${r['undervalued']:.2f}",'delta':f"{(r['undervalued']/r['price']-1)*100:.1f}%",'delta_type':'positive' if r['undervalued']<r['price'] else 'negative'},
                        {'label':'Sobrevalorada','value':f"${r['overvalued']:.2f}",'delta':f"{(r['overvalued']/r['price']-1)*100:+.1f}%",'delta_type':'positive' if r['overvalued']>r['price'] else 'negative'},
                        {'label':'CAGR Dividendo','value':f"{r['cagr']:.1f}%",'delta':'crecimiento anual','delta_type':'positive' if r['cagr']>0 else 'negative'},
                    ])

                    st.markdown("")
                    render_quality(r['quality'])
                    if r['quality']['grade'] in ['D','F']:
                        st.markdown('<div class="insight-box gold">⚠️ <strong>Quality bajo.</strong> Este ticker puede no ser ideal para el método Weiss. Funciona mejor con aristócratas de dividendos estables.</div>', unsafe_allow_html=True)

                    st.markdown(""); st.divider()

                    t_val, t_yld, t_bt, t_proj, t_div = st.tabs(["📈 Valoración","📊 Yield","🔄 Backtest","💰 Proyección","📋 Historial"])

                    with t_val:
                        st.plotly_chart(plot_valuation(r['analysis_df'], ticker.upper()), use_container_width=True)
                        c1, c2 = st.columns(2)
                        c1.markdown('<div class="insight-box green"><strong>🟢 Zona Infravalorada</strong> — Alto yield → Acción barata históricamente → Considerar compra</div>', unsafe_allow_html=True)
                        c2.markdown('<div class="insight-box red"><strong>🔴 Zona Sobrevalorada</strong> — Bajo yield → Acción cara históricamente → Considerar venta</div>', unsafe_allow_html=True)

                    with t_yld:
                        st.plotly_chart(plot_yield(r['analysis_df'], ticker.upper()), use_container_width=True)
                        st.caption("Yield TTM con percentiles históricos. P95 = zona de compra, P5 = zona de venta.")

                    with t_bt:
                        bt = r['backtest']
                        if bt is None:
                            st.info("Insuficientes señales para backtest en este período.")
                        else:
                            s = bt['stats']
                            render_metrics([
                                {'label':'Trades cerrados','value':str(s['closed_trades'])},
                                {'label':'Win rate','value':f"{s['win_rate']:.0f}%",'delta_type':'positive' if s['win_rate']>50 else 'negative'},
                                {'label':'Retorno medio','value':f"{s['avg_return']:.1f}%",'delta_type':'positive' if s['avg_return']>0 else 'negative'},
                                {'label':'Acumulado','value':f"{s['cumulative_return']:.1f}%",'delta_type':'positive' if s['cumulative_return']>0 else 'negative'},
                                {'label':'Días medios','value':f"{s['avg_holding_days']:.0f}"},
                            ])
                            if s['open_trades']>0: st.caption(f"ℹ️ {s['open_trades']} posición(es) abierta(s)")
                            fig_bt = plot_backtest_chart(r['analysis_df'], bt, ticker.upper())
                            if fig_bt: st.plotly_chart(fig_bt, use_container_width=True)
                            with st.expander("📋 Detalle"):
                                td = bt['trades'].copy()
                                td['entry_date'] = td['entry_date'].dt.strftime('%Y-%m-%d')
                                td['exit_date'] = td['exit_date'].dt.strftime('%Y-%m-%d')
                                for c in ['entry_price','exit_price','return_pct']: td[c] = td[c].round(2)
                                sc = [c for c in ['entry_date','entry_price','exit_date','exit_price','return_pct','holding_days'] if c in td.columns]
                                td = td[sc]; td.columns = ['Compra','Precio C.','Venta','Precio V.','Ret. %','Días']
                                st.dataframe(td, use_container_width=True, hide_index=True)
                            st.markdown('<div class="disclaimer">⚠️ Backtest simplificado sin comisiones, slippage ni dividendos reinvertidos. Resultados pasados no garantizan rendimientos futuros.</div>', unsafe_allow_html=True)

                    with t_proj:
                        st.markdown("#### 💰 Proyección de Ingresos por Dividendos")
                        inv = st.number_input("Inversión inicial ($)", min_value=1000, max_value=1000000, value=10000, step=1000)
                        proj = GeraldineWeissAnalyzer(ticker.upper(), years).project_dividend_income(
                            r['ttm_dividend'], r['cagr'], r['price'], inv)
                        if proj:
                            render_projection(proj, inv)
                            st.markdown('<div class="disclaimer">Proyección basada en CAGR histórico de dividendos. No es garantía de rendimiento futuro.</div>', unsafe_allow_html=True)

                    with t_div:
                        dd = r['dividend_data'].copy().sort_values('ex_dividend_date', ascending=False)
                        st.plotly_chart(plot_dividend_bars(dd), use_container_width=True)
                        ds = dd[['ex_dividend_date','amount']].copy()
                        ds['ex_dividend_date'] = ds['ex_dividend_date'].dt.strftime('%Y-%m-%d')
                        ds['amount'] = ds['amount'].round(4)
                        ds.columns = ['Fecha Ex-Div','Importe']
                        st.dataframe(ds, use_container_width=True, hide_index=True, height=350)
            else:
                st.markdown("""
                <div style="text-align:center;padding:80px 40px;color:var(--text-secondary)">
                    <div style="font-size:3rem;margin-bottom:16px">💎</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text-primary);margin-bottom:8px">
                        Selecciona un ticker para comenzar
                    </div>
                    <div style="font-size:0.95rem;max-width:400px;margin:0 auto;line-height:1.6">
                        Introduce un símbolo en la barra lateral y pulsa <strong>Analizar</strong> para obtener
                        valoración, quality score, backtest y proyección de dividendos.
                    </div>
                </div>""", unsafe_allow_html=True)

    # ==================== TAB 2 ====================
    with tab2:
        st.markdown("#### 📊 Comparación Multi-Ticker")
        c1, c2 = st.columns([2,1])
        with c1: ti = st.text_input("Tickers (comas)", value="KO, PG, JNJ, PEP")
        with c2: yc = st.selectbox("Período", [3,5,6,10], index=2); cb = st.button("🔍 Comparar", type="primary", use_container_width=True)
        if cb:
            tl = [t.strip().upper() for t in ti.split(',') if t.strip()]
            if len(tl)<2: st.error("Mínimo 2 tickers")
            else:
                if len(tl)>6: tl=tl[:6]
                with st.spinner(f'Analizando {len(tl)} tickers...'):
                    results=[]; pb=st.progress(0)
                    for i,t in enumerate(tl):
                        r=analyze_ticker_quick(t,yc)
                        if r: results.append(r)
                        pb.progress((i+1)/len(tl))
                    pb.empty(); fl=[t for t in tl if t not in [r['ticker'] for r in results]]
                    if fl: st.warning(f"⚠️ Sin datos: {', '.join(fl)}")
                    if not results: st.error("Sin datos")
                    else:
                        st.plotly_chart(plot_comparison(results), use_container_width=True)
                        st.divider()
                        cdf = pd.DataFrame([{
                            'Ticker':r['ticker'],'Precio':f"${r['price']:.2f}",
                            'Yield':f"{r['yield']:.2f}%",'Señal':r['signal'],
                            'Score':f"{r['score']:.1f}",'Quality':r['quality']['grade'],
                            'Confianza':r['confidence_label'],'CAGR':f"{r['cagr']:.1f}%"
                        } for r in results]).sort_values('Score', ascending=False)
                        st.dataframe(cdf, use_container_width=True, hide_index=True)
                        st.divider()
                        c1,c2,c3 = st.columns(3)
                        buy = sorted([r for r in results if 'COMPRA' in r['signal']], key=lambda x:x['score'], reverse=True)
                        hold = [r for r in results if r['signal']=='MANTENER']
                        sell = sorted([r for r in results if 'VENTA' in r['signal']], key=lambda x:x['score'])
                        with c1:
                            st.markdown('<div class="insight-box green"><strong>🟢 Compra</strong></div>', unsafe_allow_html=True)
                            for r in buy[:3]: st.markdown(f"**{r['ticker']}** ({r['quality']['grade']}) — Score {r['score']:.0f}")
                            if not buy: st.caption("—")
                        with c2:
                            st.markdown('<div class="insight-box gold"><strong>🟡 Mantener</strong></div>', unsafe_allow_html=True)
                            for r in hold[:3]: st.markdown(f"**{r['ticker']}** ({r['quality']['grade']})")
                            if not hold: st.caption("—")
                        with c3:
                            st.markdown('<div class="insight-box red"><strong>🔴 Venta</strong></div>', unsafe_allow_html=True)
                            for r in sell[:3]: st.markdown(f"**{r['ticker']}** ({r['quality']['grade']}) — Score {r['score']:.0f}")
                            if not sell: st.caption("—")

    # ==================== TAB 3 ====================
    with tab3:
        st.markdown("#### 💼 Cartera Ponderada")
        if 'portfolio' not in st.session_state: st.session_state.portfolio = pd.DataFrame(columns=['ticker','weight'])
        c1, c2 = st.columns([2,1])
        with c1:
            ca,cb,cc = st.columns([2,1,1])
            with ca: nt = st.text_input("Ticker", key="pt")
            with cb: nw = st.number_input("Peso %", 0.0, 100.0, 10.0, 5.0, key="pw")
            with cc:
                st.write(""); st.write("")
                if st.button("➕", use_container_width=True):
                    if nt:
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, pd.DataFrame([{'ticker':nt.upper(),'weight':nw}])], ignore_index=True); st.rerun()
        with c2:
            cc1,cc2 = st.columns(2)
            with cc1:
                if st.button("🗑️ Limpiar", use_container_width=True): st.session_state.portfolio = pd.DataFrame(columns=['ticker','weight']); st.rerun()
            with cc2:
                if st.button("📋 Demo", use_container_width=True):
                    st.session_state.portfolio = pd.DataFrame([{'ticker':'KO','weight':25},{'ticker':'JNJ','weight':25},{'ticker':'PG','weight':25},{'ticker':'PEP','weight':25}]); st.rerun()
            if not st.session_state.portfolio.empty:
                st.download_button("💾 Guardar JSON", portfolio_to_json(st.session_state.portfolio), "cartera.json", "application/json", use_container_width=True)
            uf = st.file_uploader("📂 Cargar", type=['json'], key="pu", label_visibility="collapsed")
            if uf:
                loaded = json_to_portfolio(uf.read().decode('utf-8'))
                if loaded is not None: st.session_state.portfolio = loaded; st.rerun()

        st.divider()
        if not st.session_state.portfolio.empty:
            edf = st.data_editor(st.session_state.portfolio, use_container_width=True, hide_index=True, num_rows="dynamic")
            edf = edf.dropna(subset=['ticker']); edf = edf[edf['ticker'].astype(str).str.strip()!='']
            edf['weight'] = pd.to_numeric(edf['weight'], errors='coerce').fillna(0); edf = edf[edf['weight']>0]
            st.session_state.portfolio = edf.reset_index(drop=True)
            st.divider()
            if st.button("🔍 Analizar Cartera", type="primary", use_container_width=True):
                with st.spinner('Analizando...'):
                    pd_ = st.session_state.portfolio.copy()
                    if pd_.empty: st.error("Vacía")
                    else:
                        pd_['weight'] = (pd_['weight']/pd_['weight'].sum())*100
                        pr,fl=[],[];pb=st.progress(0)
                        for i,row in pd_.iterrows():
                            r=analyze_ticker_quick(row['ticker'],6)
                            if r: r['portfolio_weight']=row['weight'];pr.append(r)
                            else: fl.append(row['ticker'])
                            pb.progress((i+1)/len(pd_))
                        pb.empty()
                        if fl: st.warning(f"⚠️ Sin datos: {', '.join(fl)}")
                        if not pr: st.error("Sin datos")
                        else:
                            ty = sum(r['yield']*r['portfolio_weight']/100 for r in pr)
                            tc = sum(r['cagr']*r['portfolio_weight']/100 for r in pr)
                            asc = sum(r['score']*r['portfolio_weight']/100 for r in pr)
                            aq = sum(r['quality']['total_score']*r['portfolio_weight']/100 for r in pr)
                            if asc>30: ps,pcol = "COMPRA","var(--accent-emerald)"
                            elif asc<-30: ps,pcol = "VENTA","var(--accent-red)"
                            else: ps,pcol = "MANTENER","var(--accent-gold)"
                            cls = 'buy' if 'COMPRA' in ps else 'sell' if 'VENTA' in ps else 'hold'
                            st.markdown(f"""<div class="signal-card {cls}"><div class="signal-glow"></div><div class="signal-label">Señal de cartera</div><div class="signal-value" style="color:{pcol}">{ps}</div><div class="signal-price">Score ponderado: {asc:.1f}</div></div>""", unsafe_allow_html=True)
                            render_metrics([
                                {'label':'Yield pond.','value':f"{ty:.2f}%"},
                                {'label':'CAGR pond.','value':f"{tc:.2f}%"},
                                {'label':'Score','value':f"{asc:.1f}"},
                                {'label':'Quality','value':f"{aq:.0f}/100"},
                                {'label':'Posiciones','value':str(len(pr))},
                            ])
                            st.divider()
                            c1,c2 = st.columns([1,1])
                            with c1: st.plotly_chart(plot_pie(pd_), use_container_width=True)
                            with c2:
                                det = pd.DataFrame([{'Ticker':r['ticker'],'Peso':f"{r['portfolio_weight']:.1f}%",'Yield':f"{r['yield']:.2f}%",'Señal':r['signal'],'Quality':r['quality']['grade'],'Score':f"{r['score']:.1f}"} for r in pr])
                                st.dataframe(det, use_container_width=True, hide_index=True)
                            st.divider()
                            ppdf = create_weighted_portfolio_analysis(pr)
                            st.plotly_chart(plot_portfolio_chart(ppdf), use_container_width=True)
                            st.divider()
                            bp = [r for r in pr if 'COMPRA' in r['signal']]; sp = [r for r in pr if 'VENTA' in r['signal']]
                            c1,c2 = st.columns(2)
                            with c1:
                                st.markdown('<div class="insight-box green"><strong>🟢 Aumentar posición</strong></div>', unsafe_allow_html=True)
                                for r in bp: st.markdown(f"**{r['ticker']}** ({r['portfolio_weight']:.1f}%) — {r['signal']} [{r['quality']['grade']}]")
                                if not bp: st.caption("—")
                            with c2:
                                st.markdown('<div class="insight-box red"><strong>🔴 Reducir posición</strong></div>', unsafe_allow_html=True)
                                for r in sp: st.markdown(f"**{r['ticker']}** ({r['portfolio_weight']:.1f}%) — {r['signal']} [{r['quality']['grade']}]")
                                if not sp: st.caption("—")
        else:
            st.markdown("""
            <div style="text-align:center;padding:60px;color:var(--text-secondary)">
                <div style="font-size:2.5rem;margin-bottom:12px">💼</div>
                <div style="font-family:'Outfit',sans-serif;font-size:1.2rem;color:var(--text-primary);margin-bottom:8px">Añade tickers para construir tu cartera</div>
                <div style="font-size:0.9rem">Usa <strong>Demo</strong> para un ejemplo rápido o <strong>Cargar</strong> para importar un JSON guardado</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="gw-footer">Desarrollado por <a href="https://bquantfinance.com" target="_blank">@Gsnchez · bquantfinance.com</a></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
