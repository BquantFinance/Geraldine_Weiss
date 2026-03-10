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
st.set_page_config(page_title="Geraldine Weiss | Dividend Intelligence", page_icon="💎", layout="wide", initial_sidebar_state="expanded")

# ════════════════════════════════════════════════════════════════
# CSS
# ════════════════════════════════════════════════════════════════

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{--b0:#05080e;--b1:#0a0e18;--b2:#0e1422;--b3:#131b2e;--b4:#182340;--a:#00d87a;--a2:#00a8cc;--gd:#e5a910;--rd:#e53e3e;--t:#dce4f0;--t2:#8899b4;--t3:#4e5e78;--br:rgba(255,255,255,0.05);--br2:rgba(255,255,255,0.09);--r:10px}

/* Base */
.main{background:var(--b1)!important}
.block-container{padding:1.2rem 2rem 4rem!important;max-width:1380px!important}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif!important;color:var(--t)}
h1,h2,h3,h4,h5,h6{font-family:'Outfit',sans-serif!important;font-weight:700!important;letter-spacing:-0.025em}
p,li,span{font-size:14px}hr{border-color:var(--br)!important;margin:1.2rem 0!important}a{color:var(--a)!important}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:var(--b0)}::-webkit-scrollbar-thumb{background:var(--b4);border-radius:4px}

/* Sidebar */
section[data-testid="stSidebar"]{background:var(--b0)!important;border-right:1px solid var(--br)!important}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:var(--b2);padding:5px;border-radius:var(--r);border:1px solid var(--br);gap:3px}
.stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:7px;padding:8px 18px;font-family:'Outfit',sans-serif;font-weight:500;font-size:13px;color:var(--t3)}
.stTabs [aria-selected="true"]{background:var(--b3)!important;color:var(--a)!important;box-shadow:0 1px 6px rgba(0,216,122,0.08)}
.stTabs [data-baseweb="tab-highlight"]{background-color:transparent!important}
.stTabs [data-baseweb="tab-border"]{display:none!important}

/* Metrics — equal columns, no truncation */
[data-testid="stMetricValue"]{font-family:'JetBrains Mono',monospace!important;font-size:22px!important;font-weight:600!important;color:var(--t)!important;letter-spacing:-0.01em}
[data-testid="stMetricLabel"]{font-family:'Outfit',sans-serif!important;font-size:10.5px!important;text-transform:uppercase;letter-spacing:0.08em;color:var(--t3)!important;white-space:nowrap!important;overflow:visible!important;text-overflow:unset!important}
[data-testid="stMetricDelta"]{font-family:'JetBrains Mono',monospace!important;font-size:11px!important}
[data-testid="stMetric"],[data-testid="metric-container"]{background:linear-gradient(145deg,var(--b3),var(--b2))!important;border:1px solid var(--br2)!important;border-radius:var(--r)!important;padding:16px!important;box-shadow:0 2px 10px rgba(0,0,0,0.25)}

/* Buttons */
.stButton>button{font-family:'Outfit',sans-serif!important;font-weight:600;border-radius:var(--r)!important;border:1px solid var(--br2)!important}
.stButton>button[kind="primary"]{background:linear-gradient(135deg,var(--a),var(--a2))!important;color:var(--b0)!important;border:none!important;font-weight:700;box-shadow:0 2px 14px rgba(0,216,122,0.18)}
.stButton>button[kind="primary"]:hover{box-shadow:0 4px 28px rgba(0,216,122,0.35)!important}
.stButton>button[kind="secondary"]{background:var(--b3)!important;color:var(--t2)!important}

/* Inputs */
.stTextInput input,.stNumberInput input{background:var(--b2)!important;border:1px solid var(--br2)!important;border-radius:var(--r)!important;color:var(--t)!important;font-family:'DM Sans',sans-serif!important}
.stTextInput input:focus,.stNumberInput input:focus{border-color:var(--a)!important;box-shadow:0 0 0 2px rgba(0,216,122,0.12)!important}
.stSelectbox>div>div{background:var(--b2)!important;border:1px solid var(--br2)!important;border-radius:var(--r)!important}

/* Slider */
.stSlider [data-baseweb="slider"] [role="slider"]{background:var(--a)!important;border-color:var(--a)!important}

/* DataFrames */
[data-testid="stDataFrame"]{border:1px solid var(--br2)!important;border-radius:var(--r)!important;overflow:hidden}

/* Expander */
details{background:var(--b2)!important;border:1px solid var(--br)!important;border-radius:var(--r)!important}
details summary{font-family:'Outfit',sans-serif!important;font-weight:500;font-size:13px;color:var(--t2)}

/* Progress */
.stProgress>div>div{background:var(--b2)!important;border-radius:4px}
.stProgress>div>div>div{background:linear-gradient(90deg,var(--a),var(--a2))!important}

/* Alerts */
[data-testid="stAlert"]{background:var(--b3)!important;border:1px solid var(--br2)!important;border-radius:var(--r)!important;color:var(--t2)!important}

/* ═══ CUSTOM COMPONENTS ═══ */

/* Hero */
.hero{padding:0.3rem 0 0.6rem}
.hero h1{font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:800;letter-spacing:-0.03em;margin:0;line-height:1.15;background:linear-gradient(135deg,var(--t) 10%,var(--a) 55%,var(--a2) 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero p{font-family:'DM Sans',sans-serif;font-size:12.5px;color:var(--t3);margin:3px 0 0}
.hero-line{height:1.5px;background:linear-gradient(90deg,var(--a),var(--a2) 35%,transparent 80%);margin-top:10px}

/* Signal */
.sig{position:relative;text-align:center;padding:28px 20px 22px;border-radius:12px;margin:14px 0;overflow:hidden;background:linear-gradient(145deg,var(--b3),var(--b2));border:1px solid var(--br2);box-shadow:0 4px 20px rgba(0,0,0,0.3)}
.sig::after{content:'';position:absolute;top:0;left:0;right:0;height:2.5px}
.sig.buy::after{background:linear-gradient(90deg,var(--a),var(--a2))}.sig.sell::after{background:linear-gradient(90deg,var(--rd),#f97316)}.sig.hold::after{background:linear-gradient(90deg,var(--gd),#f59e0b)}
.sig .sl{font-family:'Outfit',sans-serif;font-size:9.5px;font-weight:600;text-transform:uppercase;letter-spacing:0.2em;color:var(--t3);margin-bottom:5px}
.sig .sv{font-family:'Outfit',sans-serif;font-size:2rem;font-weight:900;letter-spacing:-0.02em;line-height:1.1}
.sig .ss{font-family:'JetBrains Mono',monospace;font-size:12.5px;color:var(--t2);margin-top:8px}
.sig .sg{position:absolute;top:-80px;left:50%;transform:translateX(-50%);width:240px;height:240px;border-radius:50%;filter:blur(65px);opacity:0.1;pointer-events:none}
.sig.buy .sg{background:var(--a)}.sig.sell .sg{background:var(--rd)}.sig.hold .sg{background:var(--gd)}

/* Quality */
.qg{display:flex;align-items:center;gap:16px;background:linear-gradient(145deg,var(--b3),var(--b2));border:1px solid var(--br2);border-radius:12px;padding:16px 20px;margin:10px 0;box-shadow:0 2px 10px rgba(0,0,0,0.2)}
.qg-g{width:54px;height:54px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-family:'Outfit',sans-serif;font-size:1.5rem;font-weight:900;flex-shrink:0;background:var(--b0)}
.qg-bars{flex:1;display:grid;grid-template-columns:repeat(4,1fr);gap:8px}
.qg-i{text-align:center}.qg-i .ql{font-size:9px;text-transform:uppercase;letter-spacing:0.1em;color:var(--t3);font-family:'Outfit',sans-serif;margin-bottom:4px}
.qg-tk{height:4px;background:var(--b0);border-radius:2px;overflow:hidden;margin-bottom:3px}
.qg-fl{height:100%;border-radius:2px}
.qg-i .qv{font-family:'JetBrains Mono',monospace;font-size:10.5px;color:var(--t2)}
.qg-nfo{font-size:11.5px;color:var(--t3);font-family:'DM Sans',sans-serif;margin-bottom:5px}.qg-nfo b{color:var(--t2)}

/* Badges */
.br{display:flex;gap:5px;flex-wrap:wrap;margin:4px 0 10px}
.bd{display:inline-flex;align-items:center;padding:2.5px 8px;border-radius:20px;font-family:'Outfit',sans-serif;font-size:10px;font-weight:600;letter-spacing:0.02em;border:1px solid}
.bd-g{background:rgba(0,216,122,0.08);color:var(--a);border-color:rgba(0,216,122,0.15)}
.bd-c{background:rgba(0,168,204,0.08);color:var(--a2);border-color:rgba(0,168,204,0.15)}
.bd-y{background:rgba(229,169,16,0.08);color:var(--gd);border-color:rgba(229,169,16,0.15)}
.bd-r{background:rgba(229,62,62,0.08);color:var(--rd);border-color:rgba(229,62,62,0.15)}

/* Insight */
.ins{background:var(--b3);border:1px solid var(--br);border-left:3px solid;border-radius:0 var(--r) var(--r) 0;padding:12px 16px;margin:5px 0;font-size:12.5px;color:var(--t2);line-height:1.5}
.ins.g{border-left-color:var(--a)}.ins.r{border-left-color:var(--rd)}.ins.y{border-left-color:var(--gd)}
.ins b{color:var(--t)}

/* Projection */
.pj{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:7px;margin:8px 0}
.pj-i{background:linear-gradient(145deg,var(--b3),var(--b2));border:1px solid var(--br2);border-radius:var(--r);padding:10px;text-align:center;box-shadow:0 1px 6px rgba(0,0,0,0.15)}
.pj-i .py{font-family:'Outfit',sans-serif;font-size:9.5px;color:var(--t3);text-transform:uppercase;letter-spacing:0.08em}
.pj-i .pv{font-family:'JetBrains Mono',monospace;font-size:0.95rem;font-weight:600;color:var(--a);margin-top:2px}
.pj-i .pc{font-size:10px;color:var(--t3);margin-top:1px}

/* Misc */
.disc{font-size:10px;color:var(--t3);font-style:italic;padding:8px 12px;background:var(--b2);border-radius:var(--r);border:1px solid var(--br);margin-top:8px}
.ft{position:fixed;bottom:10px;right:10px;background:rgba(10,14,24,0.85);backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);padding:5px 12px;border-radius:var(--r);border:1px solid var(--br);z-index:999;font-size:10.5px;color:var(--t3);font-family:'DM Sans',sans-serif}.ft a{color:var(--a)!important;text-decoration:none;font-weight:600}
.mt{text-align:center;padding:50px 30px}.mt .mi{font-size:2.2rem;margin-bottom:10px}.mt .mh{font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;color:var(--t);margin-bottom:5px}.mt .ms{font-size:12.5px;color:var(--t3);max-width:350px;margin:0 auto;line-height:1.5}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════════

class DividendDataFetcher:
    def __init__(self):
        self.base_url="https://dividendhistory.org/payout"; self.session=requests.Session(); self.cache={}
    def fetch_dividends(self,ticker,sd=None,ed=None):
        ck=f"{ticker}_{sd}_{ed}"
        if ck in self.cache: return self.cache[ck].copy()
        try:
            r=self.session.get(f"{self.base_url}/{ticker}/",headers={'User-Agent':'Mozilla/5.0','Accept':'text/html'},timeout=20,allow_redirects=True); r.raise_for_status()
            if len(r.text)<100: return pd.DataFrame()
            try: tables=pd.read_html(StringIO(r.text))
            except: return pd.DataFrame()
            df=None
            for t in (tables or []):
                tmp=t.copy(); tmp.columns=[str(c).strip() for c in tmp.columns]
                if all(c.isdigit() for c in tmp.columns): tmp.columns=tmp.iloc[0].astype(str).str.strip().tolist(); tmp=tmp.iloc[1:].reset_index(drop=True)
                if 'Ex-Dividend Date' in tmp.columns or 'Cash Amount' in tmp.columns: df=tmp; break
            if df is None: return pd.DataFrame()
            df=df.rename(columns={'Ex-Dividend Date':'ex_dividend_date','Payout Date':'payout_date','Cash Amount':'amount','% Change':'pct_change'})
            if 'ex_dividend_date' not in df.columns: return pd.DataFrame()
            df['ex_dividend_date']=pd.to_datetime(df['ex_dividend_date'],errors='coerce')
            if 'amount' in df.columns:
                df['amount']=df['amount'].astype(str).str.replace('$','',regex=False).str.replace(',','',regex=False).str.strip()
                df['amount']=pd.to_numeric(df['amount'],errors='coerce')
            df=df[df['ex_dividend_date'].notna()&df['amount'].notna()&(df['amount']>0)]
            if sd: df=df[df['ex_dividend_date']>=pd.to_datetime(sd)]
            if ed: df=df[df['ex_dividend_date']<=pd.to_datetime(ed)]
            df=df.sort_values('ex_dividend_date',ascending=False).reset_index(drop=True)
            if not df.empty: self.cache[ck]=df.copy()
            return df
        except: return pd.DataFrame()


class GeraldineWeissAnalyzer:
    def __init__(self,ticker,years=6):
        self.ticker,self.years=ticker,years; self.dividend_fetcher=DividendDataFetcher(); self.data_source=None

    def fetch_price_data(self):
        try:
            d=yf.Ticker(self.ticker).history(start=datetime.now()-relativedelta(years=self.years),end=datetime.now(),auto_adjust=False,actions=False)
            if d.empty: return None
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            d.index=pd.to_datetime(d.index)
            if d.index.tz is not None: d.index=d.index.tz_localize(None)
            return d
        except: return None

    def fetch_dividend_data(self):
        sd=datetime.now()-relativedelta(years=self.years)
        if '.' not in self.ticker:
            try:
                df=self.dividend_fetcher.fetch_dividends(self.ticker,sd.strftime('%Y-%m-%d'),datetime.now().strftime('%Y-%m-%d'))
                if not df.empty: self.data_source="dividendhistory.org"; return df
            except: pass
        df=self._yf(sd)
        if not df.empty: self.data_source="yfinance"; return df
        self.data_source="none"; return pd.DataFrame()

    def _yf(self,sd):
        try:
            dv=yf.Ticker(self.ticker).dividends
            if dv.empty:
                try:
                    a=yf.Ticker(self.ticker).actions
                    if 'Dividends' in a.columns: dv=a['Dividends'][a['Dividends']>0]
                except: pass
            if not dv.empty:
                if dv.index.tz: sd=(pytz.UTC.localize(sd) if sd.tzinfo is None else sd).astimezone(dv.index.tz)
                elif sd.tzinfo: sd=sd.replace(tzinfo=None)
                dv=dv[dv.index>=sd]
                if not dv.empty:
                    return pd.DataFrame({'ex_dividend_date':dv.index.tz_localize(None) if dv.index.tz else dv.index,'amount':dv.values}).query('amount>0').sort_values('ex_dividend_date',ascending=False).reset_index(drop=True)
            return pd.DataFrame()
        except: return pd.DataFrame()

    def calculate_ttm_dividends(self,prices,ddf):
        if ddf.empty or prices is None or prices.empty: return None
        p=prices.copy()
        if p.index.tz is not None: p.index=p.index.tz_localize(None)
        if 'Close' not in p.columns:
            if 'close' in p.columns: p=p.rename(columns={'close':'Close'})
            else: return None
        ds=ddf.copy().sort_values('ex_dividend_date').set_index('ex_dividend_date')['amount']
        ds.index=pd.to_datetime(ds.index)
        if ds.index.tz: ds.index=ds.index.tz_localize(None)
        m,s=ds.mean(),ds.std()
        if s>0: ds=ds[abs(ds-m)<=2.5*s]
        dd,da=ds.index,ds.values
        p['ttm_dividend']=[da[((dd>d-timedelta(days=365))&(dd<=d))].sum() for d in p.index]
        p=p[p['ttm_dividend']>0]
        return p if not p.empty else None

    def calculate_valuation_bands(self,ptm):
        if ptm is None or ptm.empty: return None
        m=ptm.copy(); m['div_yield']=m['ttm_dividend']/m['Close']
        ym,ys=m['div_yield'].median(),m['div_yield'].std()
        if ys>0: m=m[abs(m['div_yield']-ym)<=3*ys]
        if m.empty or len(m)<10: return None
        mx,mn=m['div_yield'].quantile(0.95),m['div_yield'].quantile(0.05)
        if mx<=0 or mn<=0 or mx<=mn: return None
        m['undervalued_raw']=(m['div_yield']/mx)*m['Close']
        m['overvalued_raw']=(m['div_yield']/mn)*m['Close']
        # Aggressive smoothing: rolling median → EWM → second EWM
        w=min(31,max(7,len(m)//40))
        for col in ['undervalued','overvalued']:
            raw=m[f'{col}_raw']
            s1=raw.rolling(window=w,center=True,min_periods=1).median()
            s2=s1.ewm(span=w*2,adjust=False).mean()
            m[col]=s2.ewm(span=w,adjust=False).mean()
        return m

    def get_current_signal(self,adf):
        if adf is None or adf.empty: return "DESCONOCIDO","",0
        l=adf.iloc[-1]; p,uv,ov=l['Close'],l['undervalued'],l['overvalued']; rs=ov-uv
        if rs<=0: return "DESCONOCIDO","",0
        bz,sz=uv+rs*0.2,ov-rs*0.2; sc=((ov-p)/rs)*200-100
        if p<=bz: return "COMPRA FUERTE","En zona infravalorada",sc
        elif p<=uv: return "COMPRA","Cerca de infravalorada",sc
        elif p>=sz: return "VENTA FUERTE","En zona sobrevalorada",sc
        elif p>=ov: return "VENTA","Cerca de sobrevalorada",sc
        return "MANTENER","En valor razonable",sc

    def calculate_quality_score(self,ddf):
        E={'total_score':0,'grade':'F','details':{'consecutive_years':0,'n_years':0,'years_score':0,'growth_pct':0,'growth_score':0,'cv':0,'stability_score':0,'payments_per_year':0,'frequency_score':0}}
        if ddf.empty: return E
        d=ddf.copy().sort_values('ex_dividend_date'); d['year']=d['ex_dividend_date'].dt.year
        uy=sorted(d['year'].unique()); ny=len(uy); c_,mx_=1,1
        for i in range(1,len(uy)):
            if uy[i]==uy[i-1]+1: c_+=1; mx_=max(mx_,c_)
            else: c_=1
        ys_=min(30,mx_*5); ann=d.groupby('year')['amount'].sum().sort_index()
        if len(ann)>=2: gy=sum(1 for i in range(1,len(ann)) if ann.iloc[i]>=ann.iloc[i-1]); gp=gy/(len(ann)-1); gs_=int(gp*30)
        else: gp,gs_=0,0
        if len(ann)>=2 and ann.mean()>0:
            cv=ann.std()/ann.mean(); ss_=20 if cv<0.1 else 15 if cv<0.2 else 10 if cv<0.35 else 5 if cv<0.5 else 0
        else: cv,ss_=0,0
        ppy=len(d)/max(ny,1); fs_=20 if ppy>=3.5 else 15 if ppy>=1.8 else 10 if ppy>=0.9 else 5
        tot=ys_+gs_+ss_+fs_; gr='A' if tot>=80 else 'B' if tot>=60 else 'C' if tot>=40 else 'D' if tot>=20 else 'F'
        return {'total_score':tot,'grade':gr,'details':{'consecutive_years':mx_,'n_years':ny,'years_score':ys_,'growth_pct':gp*100,'growth_score':gs_,'cv':cv,'stability_score':ss_,'payments_per_year':ppy,'frequency_score':fs_}}

    def calculate_confidence(self,ddf,adf):
        if ddf.empty or adf is None or adf.empty: return 'low','Baja'
        s=(3 if len(ddf)>=20 else 2 if len(ddf)>=12 else 1 if len(ddf)>=6 else 0)
        s+=(3 if ddf['ex_dividend_date'].dt.year.nunique()>=5 else 2 if ddf['ex_dividend_date'].dt.year.nunique()>=3 else 1 if ddf['ex_dividend_date'].dt.year.nunique()>=2 else 0)
        s+=(2 if len(adf)>=1000 else 1 if len(adf)>=500 else 0)
        return ('high','Alta') if s>=7 else ('medium','Media') if s>=4 else ('low','Baja')

    def backtest_signals(self,adf):
        if adf is None or adf.empty or len(adf)<50: return None
        df=adf.sort_index(); trades=[]; pos=None
        for i in range(len(df)):
            r=df.iloc[i]; d=df.index[i]; p,uv,ov=r['Close'],r['undervalued'],r['overvalued']; rs=ov-uv
            if rs<=0: continue
            bz,sz=uv+rs*0.2,ov-rs*0.2
            if pos is None and p<=bz: pos={'ed':d,'ep':p}
            elif pos and p>=sz:
                trades.append({'entry_date':pos['ed'],'entry_price':pos['ep'],'exit_date':d,'exit_price':p,'return_pct':(p/pos['ep']-1)*100,'holding_days':(d-pos['ed']).days,'open':False}); pos=None
        if pos:
            lr=df.iloc[-1]; trades.append({'entry_date':pos['ed'],'entry_price':pos['ep'],'exit_date':df.index[-1],'exit_price':lr['Close'],'return_pct':(lr['Close']/pos['ep']-1)*100,'holding_days':(df.index[-1]-pos['ed']).days,'open':True})
        if not trades: return None
        tdf=pd.DataFrame(trades); ct=tdf[~tdf['open'].astype(bool)]
        s={'total_trades':len(tdf),'closed_trades':len(ct),'open_trades':len(tdf)-len(ct)}
        if not ct.empty:
            s['win_rate']=(ct['return_pct']>0).mean()*100; s['avg_return']=ct['return_pct'].mean(); s['avg_holding_days']=ct['holding_days'].mean()
            cum=1.0
            for r_ in ct['return_pct']: cum*=(1+r_/100)
            s['cumulative_return']=(cum-1)*100
        else:
            for k in ['win_rate','avg_return','avg_holding_days','cumulative_return']: s[k]=0
        return {'trades':tdf,'stats':s}

    def project_dividend_income(self,ttm,cagr,price,inv=10000,yrs=5):
        if ttm<=0 or price<=0: return None
        sh=inv/price
        return [{'year':y,'income':sh*ttm*((1+cagr/100)**y),'yoc':(ttm*((1+cagr/100)**y)/price)*100} for y in range(yrs+1)]


@st.cache_data(ttl=3600,show_spinner=False)
def analyze(ticker,years=6):
    try:
        a=GeraldineWeissAnalyzer(ticker,years); pd_=a.fetch_price_data()
        if pd_ is None or pd_.empty: return None
        dd=a.fetch_dividend_data()
        if dd.empty: return None
        pttm=a.calculate_ttm_dividends(pd_,dd)
        if pttm is None: return None
        adf=a.calculate_valuation_bands(pttm)
        if adf is None or adf.empty: return None
        sig,desc,sc=a.get_current_signal(adf); lat=adf.iloc[-1]
        ds=dd.copy().sort_values('ex_dividend_date'); ds['year']=ds['ex_dividend_date'].dt.year
        ann=ds.groupby('year')['amount'].sum().sort_index()
        cagr=((ann.iloc[-1]/ann.iloc[0])**(1/(len(ann)-1))-1)*100 if len(ann)>1 and ann.iloc[0]>0 else 0
        q=a.calculate_quality_score(dd); cl,clb=a.calculate_confidence(dd,adf)
        return {'ticker':ticker,'price':lat['Close'],'yield':lat['div_yield']*100,'ttm_dividend':lat['ttm_dividend'],
            'undervalued':lat['undervalued'],'overvalued':lat['overvalued'],'signal':sig,'desc':desc,'score':sc,'cagr':cagr,
            'adf':adf,'dd':dd,'src':a.data_source,'quality':q,'conf':cl,'conf_label':clb,
            'bt':a.backtest_signals(adf),'proj':a.project_dividend_income(lat['ttm_dividend'],cagr,lat['Close'])}
    except: return None


# ════════════════════════════════════════════════════════════════
# PORTFOLIO
# ════════════════════════════════════════════════════════════════

def portfolio_analysis(pr):
    dates=sorted(set(d for r in pr for d in r['adf'].index.tolist()))
    pdf=pd.DataFrame(index=pd.DatetimeIndex(dates)); pdf['wp']=0.0; pdf['wu']=0.0; pdf['wo']=0.0
    for d in dates:
        tw,wp,wu,wo=0,0,0,0
        for r in pr:
            df=r['adf']
            if d in df.index: row=df.loc[d]
            else:
                av=df.index[df.index<=d]
                if len(av)>0: row=df.loc[av[-1]]
                else: continue
            if isinstance(row,pd.DataFrame): row=row.iloc[-1]
            w=r['pw']/100; tw+=w; wp+=row['Close']*w; wu+=row['undervalued']*w; wo+=row['overvalued']*w
        if tw>0: pdf.loc[d,'wp']=wp/tw; pdf.loc[d,'wu']=wu/tw; pdf.loc[d,'wo']=wo/tw
    return pdf[(pdf!=0).all(axis=1)]

def p2j(p): return json.dumps(p.to_dict(orient='records'),indent=2,ensure_ascii=False)
def j2p(j):
    try:
        df=pd.DataFrame(json.loads(j))
        if 'ticker' in df.columns and 'weight' in df.columns:
            df['weight']=pd.to_numeric(df['weight'],errors='coerce').fillna(0); return df[['ticker','weight']]
    except: pass
    return None


# ════════════════════════════════════════════════════════════════
# RENDER HELPERS
# ════════════════════════════════════════════════════════════════

def _hero():
    st.markdown('<div class="hero"><h1>Geraldine Weiss</h1><p>Dividend Intelligence — Valoración profesional por rentabilidad de dividendos</p><div class="hero-line"></div></div>',unsafe_allow_html=True)

def _signal(sig,price,desc):
    cls='buy' if 'COMPRA' in sig else 'sell' if 'VENTA' in sig else 'hold'
    cm={"COMPRA FUERTE":"var(--a)","COMPRA":"#51cf66","MANTENER":"var(--gd)","VENTA":"#f97316","VENTA FUERTE":"var(--rd)"}
    st.markdown(f'<div class="sig {cls}"><div class="sg"></div><div class="sl">Señal de valoración</div><div class="sv" style="color:{cm.get(sig,"#fff")}">{sig}</div><div class="ss">&#36;{price:.2f} · {desc}</div></div>',unsafe_allow_html=True)

def _badges(src,cl,clb):
    sc='bd-g' if src=='dividendhistory.org' else 'bd-c'
    sn='dividendhistory.org' if src=='dividendhistory.org' else 'yfinance'
    cc={'high':'bd-g','medium':'bd-y','low':'bd-r'}[cl]
    st.markdown(f'<div class="br"><span class="bd {sc}">📊 {sn}</span><span class="bd {cc}">🎯 {clb}</span></div>',unsafe_allow_html=True)

def _quality(q):
    gr,tot,d=q['grade'],q['total_score'],q['details']
    gc={'A':'var(--a)','B':'#51cf66','C':'var(--gd)','D':'#f97316','F':'var(--rd)'}.get(gr,'#fff')
    bars=[('Historial',d['years_score'],30,'var(--a)'),('Crecimiento',d['growth_score'],30,'var(--a2)'),('Estabilidad',d['stability_score'],20,'var(--gd)'),('Frecuencia',d['frequency_score'],20,'#a78bfa')]
    bh="".join(f'<div class="qg-i"><div class="ql">{l}</div><div class="qg-tk"><div class="qg-fl" style="width:{v/m*100}%;background:{c}"></div></div><div class="qv">{v}/{m}</div></div>' for l,v,m,c in bars)
    st.markdown(f'<div class="qg"><div class="qg-g" style="border:2px solid {gc};color:{gc}">{gr}</div><div style="flex:1"><div class="qg-nfo"><b>{tot}/100</b> · {d["consecutive_years"]} años · {d["payments_per_year"]:.1f} pagos/año · Crec. {d["growth_pct"]:.0f}%</div><div class="qg-bars">{bh}</div></div></div>',unsafe_allow_html=True)

def _projection(proj,inv=10000):
    if not proj: return
    items=""
    for p in proj:
        yr="Hoy" if p['year']==0 else f"Año {p['year']}"
        items+=f'<div class="pj-i"><div class="py">{yr}</div><div class="pv">&#36;{p["income"]:.0f}</div><div class="pc">YoC {p["yoc"]:.2f}%</div></div>'
    st.markdown(f'<div style="font-size:11px;color:var(--t3);margin-bottom:5px">Proyección anual · Inversión: &#36;{inv:,.0f}</div><div class="pj">{items}</div>',unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# CHARTS
# ════════════════════════════════════════════════════════════════

CC={'bg':'#0a0e18','g':'rgba(255,255,255,0.03)','t':'#8899b4','a':'#00d87a','c':'#00a8cc','r':'#e53e3e','gd':'#e5a910','ad':'rgba(0,216,122,0.07)'}

def _L(title='',h=480):
    return dict(template='plotly_dark',plot_bgcolor=CC['bg'],paper_bgcolor=CC['bg'],height=h,hovermode='x unified',margin=dict(l=50,r=20,t=55,b=40),
        title=dict(text=title,font=dict(family='Outfit',size=15,color='#dce4f0'),x=0.5,xanchor='center'),
        xaxis=dict(gridcolor=CC['g'],showgrid=True,zeroline=False,tickformat='%b %Y',tickfont=dict(family='JetBrains Mono',size=9,color=CC['t'])),
        yaxis=dict(gridcolor=CC['g'],showgrid=True,zeroline=False,tickfont=dict(family='JetBrains Mono',size=9,color=CC['t'])),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1,font=dict(family='DM Sans',size=10,color=CC['t']),bgcolor='rgba(10,14,24,0.8)',bordercolor='rgba(255,255,255,0.04)',borderwidth=1),
        hoverlabel=dict(bgcolor='#131b2e',bordercolor='rgba(255,255,255,0.06)',font=dict(family='DM Sans',size=11)))

def ch_val(adf,tk):
    a=adf.copy(); fig=go.Figure()
    fig.add_trace(go.Scatter(x=a.index,y=a['overvalued'],line=dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=a.index,y=a['undervalued'],name='Rango Valor',fill='tonexty',fillcolor=CC['ad'],line=dict(color='rgba(0,0,0,0)'),hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=a.index,y=a['overvalued'],name='Sobrevalorada',line=dict(color=CC['r'],width=1.8,dash='dot'),hovertemplate='%{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=a.index,y=a['undervalued'],name='Infravalorada',line=dict(color=CC['a'],width=1.8,dash='dot'),hovertemplate='%{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=a.index,y=a['Close'],name='Precio',line=dict(color=CC['c'],width=2.2),hovertemplate='%{y:.2f}<extra></extra>'))
    fig.add_trace(go.Scatter(x=[a.index[-1]],y=[a.iloc[-1]['Close']],mode='markers',marker=dict(size=7,color=CC['c'],line=dict(color='white',width=1.5)),showlegend=False))
    fig.update_layout(**_L(f'{tk} — Bandas de Valoración'))
    return fig

def ch_yield(adf,tk):
    a=adf.copy(); yp=a['div_yield']*100; p95,p50,p05=yp.quantile(0.95),yp.median(),yp.quantile(0.05)
    fig=go.Figure()
    fig.add_hrect(y0=p95*0.9,y1=yp.max()*1.05,fillcolor="rgba(0,216,122,0.04)",line_width=0)
    fig.add_hrect(y0=yp.min()*0.95,y1=p05*1.1,fillcolor="rgba(229,62,62,0.03)",line_width=0)
    for v,c,l in [(p95,CC['a'],f'P95 {p95:.2f}%'),(p50,CC['gd'],f'Med {p50:.2f}%'),(p05,CC['r'],f'P5 {p05:.2f}%')]:
        fig.add_hline(y=v,line=dict(color=c,width=1,dash='dash'),annotation_text=l,annotation_position="right",annotation_font=dict(color=c,size=9,family='JetBrains Mono'))
    fig.add_trace(go.Scatter(x=a.index,y=yp,line=dict(color=CC['c'],width=2),fill='tozeroy',fillcolor='rgba(0,168,204,0.05)',hovertemplate='%{y:.2f}%<extra></extra>'))
    fig.add_trace(go.Scatter(x=[a.index[-1]],y=[yp.iloc[-1]],mode='markers',marker=dict(size=7,color=CC['c'],line=dict(color='white',width=1.5)),showlegend=False))
    fig.update_layout(**_L(f'{tk} — Yield TTM',380)); fig.update_yaxes(ticksuffix='%')
    return fig

def ch_bt(adf,bt,tk):
    if bt is None: return None
    a=adf.copy(); tr=bt['trades']; fig=go.Figure()
    fig.add_trace(go.Scatter(x=a.index,y=a['Close'],name='Precio',line=dict(color='rgba(0,168,204,0.3)',width=1.5)))
    fig.add_trace(go.Scatter(x=tr['entry_date'],y=tr['entry_price'],mode='markers',name='Compra',marker=dict(size=9,color=CC['a'],symbol='triangle-up',line=dict(color='white',width=1))))
    fig.add_trace(go.Scatter(x=tr['exit_date'],y=tr['exit_price'],mode='markers',name='Venta',marker=dict(size=9,color=CC['r'],symbol='triangle-down',line=dict(color='white',width=1))))
    for _,t in tr.iterrows():
        fig.add_trace(go.Scatter(x=[t['entry_date'],t['exit_date']],y=[t['entry_price'],t['exit_price']],mode='lines',line=dict(color=CC['a'] if t['return_pct']>0 else CC['r'],width=0.8,dash='dash'),showlegend=False,hoverinfo='skip'))
    fig.update_layout(**_L(f'{tk} — Backtest',420))
    return fig

def ch_port(pdf):
    p=pdf.copy(); fig=go.Figure()
    fig.add_trace(go.Scatter(x=p.index,y=p['wo'],line=dict(color='rgba(0,0,0,0)'),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=p.index,y=p['wu'],name='Rango',fill='tonexty',fillcolor=CC['ad'],line=dict(color='rgba(0,0,0,0)'),hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=p.index,y=p['wo'],name='Sobrev.',line=dict(color=CC['r'],width=1.8,dash='dot')))
    fig.add_trace(go.Scatter(x=p.index,y=p['wu'],name='Infrav.',line=dict(color=CC['a'],width=1.8,dash='dot')))
    fig.add_trace(go.Scatter(x=p.index,y=p['wp'],name='Ponderado',line=dict(color=CC['c'],width=2.2)))
    fig.add_trace(go.Scatter(x=[p.index[-1]],y=[p.iloc[-1]['wp']],mode='markers',marker=dict(size=7,color=CC['c'],line=dict(color='white',width=1.5)),showlegend=False))
    fig.update_layout(**_L('Cartera — Valoración Ponderada'))
    return fig

def ch_comp(res):
    fig=go.Figure(); tks=[r['ticker'] for r in res]; ps=[r['price'] for r in res]; uv=[r['undervalued'] for r in res]; ov=[r['overvalued'] for r in res]; x=list(range(len(tks)))
    fig.add_trace(go.Bar(x=x,y=[o-u for o,u in zip(ov,uv)],base=uv,name='Rango',marker_color='rgba(0,216,122,0.07)',marker_line=dict(color='rgba(0,216,122,0.15)',width=1),hoverinfo='skip'))
    cols=[CC['a'] if 'COMPRA' in r['signal'] else CC['r'] if 'VENTA' in r['signal'] else CC['gd'] for r in res]
    fig.add_trace(go.Scatter(x=x,y=ps,name='Precio',mode='markers',marker=dict(size=12,color=cols,line=dict(color='white',width=1.5),symbol='diamond')))
    fig.update_layout(**_L('Comparación',400)); fig.update_xaxes(tickmode='array',tickvals=x,ticktext=tks,tickformat=None)
    return fig

def ch_pie(pd_):
    fig=go.Figure(); cols=['#00d87a','#00a8cc','#7b2ff7','#e5a910','#e53e3e','#51cf66','#f97316','#06b6d4','#a78bfa','#fb7185']
    fig.add_trace(go.Pie(labels=pd_['ticker'],values=pd_['weight'],marker=dict(colors=cols[:len(pd_)],line=dict(color=CC['bg'],width=3)),textinfo='label+percent',textfont=dict(family='Outfit',size=11,color='white'),hole=0.5))
    fig.update_layout(template='plotly_dark',height=350,plot_bgcolor=CC['bg'],paper_bgcolor=CC['bg'],showlegend=False,margin=dict(l=10,r=10,t=10,b=10),
        annotations=[dict(text='<b>Cartera</b>',font=dict(family='Outfit',size=13,color=CC['t']),showarrow=False)])
    return fig

def ch_bars(dd):
    da=dd.copy(); da['year']=dd['ex_dividend_date'].dt.year; da=da.groupby('year')['amount'].sum().sort_index()
    cols=[CC['a'] if i==len(da)-1 else 'rgba(0,216,122,0.3)' for i in range(len(da))]
    fig=go.Figure(); fig.add_trace(go.Bar(x=da.index,y=da.values,marker_color=cols,hovertemplate='%{x}: %{y:.4f}<extra></extra>'))
    fig.update_layout(**_L('Dividendo Anual',270)); fig.update_xaxes(dtick=1,tickformat='d')
    return fig


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    _hero()
    t1,t2,t3=st.tabs(["🎯  Análisis Individual","📊  Comparación","💼  Cartera"])

    with t1:
        cs,cm=st.columns([1,3])
        with cs:
            st.markdown("##### ⚙️ Configuración")
            tk=st.text_input("Ticker",value="KO"); yr=st.slider("Período (años)",3,10,6)
            st.divider()
            go_=st.button("🔍  Analizar",type="primary",use_container_width=True)
            st.divider()
            with st.expander("💡 Método Weiss"):
                st.caption("**Alto yield** → Infravalorada → Compra\n\n**Bajo yield** → Sobrevalorada → Venta\n\nUsa TTM (trailing 12m) para precisión real.")
            st.divider()
            st.caption("🇺🇸 KO · JNJ · PG · MMM · XOM")
            st.caption("🇪🇸 IBE.MC · SAN.MC · TEF.MC")
            st.caption("🇬🇧 BP.L · ULVR.L · 🇨🇦 RY.TO")

        with cm:
            if go_ and tk:
                with st.spinner('Analizando...'):
                    r=analyze(tk.upper(),yr)
                if r is None:
                    st.error(f"Sin datos para **{tk.upper()}**. Verifica ticker o dividendos.")
                else:
                    _badges(r['src'],r['conf'],r['conf_label'])
                    _signal(r['signal'],r['price'],r['desc'])

                    # Row 1: 3 metrics
                    m1,m2,m3=st.columns(3)
                    m1.metric("Precio",f"{r['price']:.2f}")
                    m2.metric("Yield TTM",f"{r['yield']:.2f}%")
                    m3.metric("CAGR Div.",f"{r['cagr']:.1f}%",delta="crecimiento anual" if r['cagr']>0 else "decrecimiento",delta_color="normal" if r['cagr']>0 else "inverse")
                    # Row 2: 3 metrics
                    m4,m5,m6=st.columns(3)
                    m4.metric("Dividendo TTM",f"{r['ttm_dividend']:.3f}")
                    m5.metric("Zona Compra",f"{r['undervalued']:.2f}",delta=f"{(r['undervalued']/r['price']-1)*100:.1f}%",delta_color="inverse")
                    m6.metric("Zona Venta",f"{r['overvalued']:.2f}",delta=f"{(r['overvalued']/r['price']-1)*100:+.1f}%")

                    st.markdown("")
                    _quality(r['quality'])
                    if r['quality']['grade'] in ['D','F']:
                        st.markdown('<div class="ins y">⚠️ <b>Quality bajo.</b> Puede no ser ideal para el método Weiss.</div>',unsafe_allow_html=True)

                    st.divider()
                    tv,ty,tb,tp,td_=st.tabs(["📈 Valoración","📊 Yield","🔄 Backtest","💰 Proyección","📋 Historial"])

                    with tv:
                        st.plotly_chart(ch_val(r['adf'],tk.upper()),use_container_width=True)
                        c1,c2=st.columns(2)
                        c1.markdown('<div class="ins g"><b>🟢 Infravalorada</b> — Alto yield → barata históricamente → Considerar compra</div>',unsafe_allow_html=True)
                        c2.markdown('<div class="ins r"><b>🔴 Sobrevalorada</b> — Bajo yield → cara históricamente → Considerar venta</div>',unsafe_allow_html=True)

                    with ty:
                        st.plotly_chart(ch_yield(r['adf'],tk.upper()),use_container_width=True)
                        st.caption("P95 = zona compra · P5 = zona venta · Mediana = valor justo")

                    with tb:
                        bt=r['bt']
                        if bt is None: st.info("Insuficientes señales para backtest.")
                        else:
                            s=bt['stats']
                            b1,b2,b3,b4=st.columns(4)
                            b1.metric("Trades",s['closed_trades']); b2.metric("Win Rate",f"{s['win_rate']:.0f}%")
                            b3.metric("Ret. Medio",f"{s['avg_return']:.1f}%"); b4.metric("Acumulado",f"{s['cumulative_return']:.1f}%")
                            if s['open_trades']>0: st.caption(f"ℹ️ {s['open_trades']} pos. abierta(s)")
                            f=ch_bt(r['adf'],bt,tk.upper())
                            if f: st.plotly_chart(f,use_container_width=True)
                            with st.expander("📋 Detalle"):
                                td__=bt['trades'].copy(); td__['entry_date']=td__['entry_date'].dt.strftime('%Y-%m-%d'); td__['exit_date']=td__['exit_date'].dt.strftime('%Y-%m-%d')
                                for c in ['entry_price','exit_price','return_pct']: td__[c]=td__[c].round(2)
                                td__=td__[[c for c in ['entry_date','entry_price','exit_date','exit_price','return_pct','holding_days'] if c in td__.columns]]
                                td__.columns=['Compra','Precio C.','Venta','Precio V.','Ret %','Días']
                                st.dataframe(td__,use_container_width=True,hide_index=True)
                            st.markdown('<div class="disc">⚠️ Backtest simplificado. Sin comisiones ni dividendos reinvertidos.</div>',unsafe_allow_html=True)

                    with tp:
                        inv=st.number_input("Inversión inicial",min_value=1000,max_value=1000000,value=10000,step=1000)
                        proj=GeraldineWeissAnalyzer(tk.upper(),yr).project_dividend_income(r['ttm_dividend'],r['cagr'],r['price'],inv)
                        if proj: _projection(proj,inv)
                        st.markdown('<div class="disc">Proyección basada en CAGR histórico. No garantiza rendimiento.</div>',unsafe_allow_html=True)

                    with td_:
                        dd=r['dd'].copy().sort_values('ex_dividend_date',ascending=False)
                        st.plotly_chart(ch_bars(dd),use_container_width=True)
                        ds=dd[['ex_dividend_date','amount']].copy(); ds['ex_dividend_date']=ds['ex_dividend_date'].dt.strftime('%Y-%m-%d'); ds['amount']=ds['amount'].round(4)
                        ds.columns=['Fecha Ex-Div','Importe']
                        st.dataframe(ds,use_container_width=True,hide_index=True,height=300)
            else:
                st.markdown('<div class="mt"><div class="mi">💎</div><div class="mh">Selecciona un ticker</div><div class="ms">Introduce un símbolo y pulsa Analizar para obtener valoración, quality score, backtest y proyección.</div></div>',unsafe_allow_html=True)

    with t2:
        st.markdown("##### 📊 Comparación")
        c1,c2=st.columns([2,1])
        with c1: ti=st.text_input("Tickers (comas)",value="KO, PG, JNJ, PEP")
        with c2: yc=st.selectbox("Período",[3,5,6,10],index=2); cb=st.button("🔍 Comparar",type="primary",use_container_width=True)
        if cb:
            tl=[t.strip().upper() for t in ti.split(',') if t.strip()][:6]
            if len(tl)<2: st.error("Mínimo 2 tickers")
            else:
                with st.spinner(f'{len(tl)} tickers...'):
                    res=[];pb=st.progress(0)
                    for i,t in enumerate(tl):
                        r=analyze(t,yc)
                        if r: res.append(r)
                        pb.progress((i+1)/len(tl))
                    pb.empty(); fl=[t for t in tl if t not in [r['ticker'] for r in res]]
                    if fl: st.warning(f"Sin datos: {', '.join(fl)}")
                    if not res: st.error("Sin datos")
                    else:
                        st.plotly_chart(ch_comp(res),use_container_width=True)
                        st.divider()
                        cdf=pd.DataFrame([{'Ticker':r['ticker'],'Precio':f"{r['price']:.2f}",'Yield':f"{r['yield']:.2f}%",'Señal':r['signal'],'Score':f"{r['score']:.1f}",'Quality':r['quality']['grade'],'CAGR':f"{r['cagr']:.1f}%"} for r in res]).sort_values('Score',ascending=False)
                        st.dataframe(cdf,use_container_width=True,hide_index=True)
                        st.divider()
                        c1,c2,c3=st.columns(3)
                        buy=sorted([r for r in res if 'COMPRA' in r['signal']],key=lambda x:x['score'],reverse=True)
                        hold=[r for r in res if r['signal']=='MANTENER']
                        sell=sorted([r for r in res if 'VENTA' in r['signal']],key=lambda x:x['score'])
                        with c1:
                            st.markdown('<div class="ins g"><b>🟢 Compra</b></div>',unsafe_allow_html=True)
                            for r in buy[:3]: st.caption(f"**{r['ticker']}** ({r['quality']['grade']}) — Score {r['score']:.0f}")
                            if not buy: st.caption("—")
                        with c2:
                            st.markdown('<div class="ins y"><b>🟡 Mantener</b></div>',unsafe_allow_html=True)
                            for r in hold[:3]: st.caption(f"**{r['ticker']}** ({r['quality']['grade']})")
                            if not hold: st.caption("—")
                        with c3:
                            st.markdown('<div class="ins r"><b>🔴 Venta</b></div>',unsafe_allow_html=True)
                            for r in sell[:3]: st.caption(f"**{r['ticker']}** ({r['quality']['grade']}) — Score {r['score']:.0f}")
                            if not sell: st.caption("—")

    with t3:
        st.markdown("##### 💼 Cartera Ponderada")
        if 'pf' not in st.session_state: st.session_state.pf=pd.DataFrame(columns=['ticker','weight'])
        c1,c2=st.columns([2,1])
        with c1:
            ca,cb,cc=st.columns([2,1,1])
            with ca: nt=st.text_input("Ticker",key="pt")
            with cb: nw=st.number_input("Peso %",0.0,100.0,10.0,5.0,key="pw")
            with cc:
                st.write(""); st.write("")
                if st.button("➕",use_container_width=True):
                    if nt: st.session_state.pf=pd.concat([st.session_state.pf,pd.DataFrame([{'ticker':nt.upper(),'weight':nw}])],ignore_index=True); st.rerun()
        with c2:
            cc1,cc2=st.columns(2)
            with cc1:
                if st.button("🗑️ Limpiar",use_container_width=True): st.session_state.pf=pd.DataFrame(columns=['ticker','weight']); st.rerun()
            with cc2:
                if st.button("📋 Demo",use_container_width=True): st.session_state.pf=pd.DataFrame([{'ticker':'KO','weight':25},{'ticker':'JNJ','weight':25},{'ticker':'PG','weight':25},{'ticker':'PEP','weight':25}]); st.rerun()
            if not st.session_state.pf.empty:
                st.download_button("💾 Guardar",p2j(st.session_state.pf),"cartera.json","application/json",use_container_width=True)
            uf=st.file_uploader("📂 Cargar",type=['json'],key="pu",label_visibility="collapsed")
            if uf:
                ld=j2p(uf.read().decode('utf-8'))
                if ld is not None: st.session_state.pf=ld; st.rerun()

        st.divider()
        if not st.session_state.pf.empty:
            edf=st.data_editor(st.session_state.pf,use_container_width=True,hide_index=True,num_rows="dynamic")
            edf=edf.dropna(subset=['ticker']); edf=edf[edf['ticker'].astype(str).str.strip()!='']
            edf['weight']=pd.to_numeric(edf['weight'],errors='coerce').fillna(0); edf=edf[edf['weight']>0]
            st.session_state.pf=edf.reset_index(drop=True)
            st.divider()
            if st.button("🔍 Analizar Cartera",type="primary",use_container_width=True):
                with st.spinner('Analizando...'):
                    pd__=st.session_state.pf.copy()
                    if pd__.empty: st.error("Vacía")
                    else:
                        pd__['weight']=(pd__['weight']/pd__['weight'].sum())*100
                        pr=[];fl=[];pb=st.progress(0)
                        for i,row in pd__.iterrows():
                            r=analyze(row['ticker'],6)
                            if r: r['pw']=row['weight'];pr.append(r)
                            else: fl.append(row['ticker'])
                            pb.progress((i+1)/len(pd__))
                        pb.empty()
                        if fl: st.warning(f"Sin datos: {', '.join(fl)}")
                        if not pr: st.error("Sin datos")
                        else:
                            ty=sum(r['yield']*r['pw']/100 for r in pr); tc=sum(r['cagr']*r['pw']/100 for r in pr)
                            asc=sum(r['score']*r['pw']/100 for r in pr); aq=sum(r['quality']['total_score']*r['pw']/100 for r in pr)
                            ps='COMPRA' if asc>30 else 'VENTA' if asc<-30 else 'MANTENER'
                            cls='buy' if 'COMPRA' in ps else 'sell' if 'VENTA' in ps else 'hold'
                            pcm={"COMPRA":"var(--a)","VENTA":"var(--rd)","MANTENER":"var(--gd)"}
                            st.markdown(f'<div class="sig {cls}"><div class="sg"></div><div class="sl">Señal de cartera</div><div class="sv" style="color:{pcm[ps]}">{ps}</div><div class="ss">Score ponderado: {asc:.1f}</div></div>',unsafe_allow_html=True)

                            m1,m2,m3=st.columns(3)
                            m1.metric("Yield Pond.",f"{ty:.2f}%"); m2.metric("CAGR Pond.",f"{tc:.2f}%"); m3.metric("Score",f"{asc:.1f}")
                            m4,m5,_=st.columns(3)
                            m4.metric("Quality",f"{aq:.0f}/100"); m5.metric("Posiciones",len(pr))
                            st.divider()
                            c1,c2=st.columns(2)
                            with c1: st.plotly_chart(ch_pie(pd__),use_container_width=True)
                            with c2:
                                det=pd.DataFrame([{'Ticker':r['ticker'],'Peso':f"{r['pw']:.1f}%",'Yield':f"{r['yield']:.2f}%",'Señal':r['signal'],'Quality':r['quality']['grade'],'Score':f"{r['score']:.1f}"} for r in pr])
                                st.dataframe(det,use_container_width=True,hide_index=True)
                            st.divider()
                            ppdf=portfolio_analysis(pr)
                            st.plotly_chart(ch_port(ppdf),use_container_width=True)
                            st.divider()
                            bp=[r for r in pr if 'COMPRA' in r['signal']]; sp=[r for r in pr if 'VENTA' in r['signal']]
                            c1,c2=st.columns(2)
                            with c1:
                                st.markdown('<div class="ins g"><b>🟢 Aumentar</b></div>',unsafe_allow_html=True)
                                for r in bp: st.caption(f"**{r['ticker']}** ({r['pw']:.0f}%) — {r['signal']} [{r['quality']['grade']}]")
                                if not bp: st.caption("—")
                            with c2:
                                st.markdown('<div class="ins r"><b>🔴 Reducir</b></div>',unsafe_allow_html=True)
                                for r in sp: st.caption(f"**{r['ticker']}** ({r['pw']:.0f}%) — {r['signal']} [{r['quality']['grade']}]")
                                if not sp: st.caption("—")
        else:
            st.markdown('<div class="mt"><div class="mi">💼</div><div class="mh">Construye tu cartera</div><div class="ms">Añade tickers con pesos o usa Demo. Puedes guardar y cargar en JSON.</div></div>',unsafe_allow_html=True)

    st.markdown('<div class="ft">by <a href="https://bquantfinance.com" target="_blank">@Gsnchez · bquantfinance.com</a></div>',unsafe_allow_html=True)

if __name__=="__main__": main()
