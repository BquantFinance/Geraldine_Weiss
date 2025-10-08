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

# Suprimir warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuración de la página
st.set_page_config(
    page_title="Geraldine Weiss | Estrategia de Dividendos",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS para tema oscuro
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 32px;
        color: #00ff88;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2839;
        padding: 12px 24px;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2d3748;
        border-bottom: 2px solid #00ff88;
    }
    
    .big-signal {
        text-align: center;
        padding: 40px;
        border-radius: 16px;
        margin: 30px 0;
        border: 3px solid;
        font-size: 48px;
        font-weight: 700;
    }
    
    .data-source-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-left: 10px;
    }
    
    .source-dividendhistory {
        background-color: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        border: 1px solid #00ff88;
    }
    
    .source-yfinance {
        background-color: rgba(0, 212, 255, 0.2);
        color: #00d4ff;
        border: 1px solid #00d4ff;
    }
    
    .footer-credit {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background-color: rgba(26, 31, 46, 0.95);
        padding: 10px 20px;
        border-radius: 8px;
        border: 1px solid #00ff88;
        z-index: 999;
        font-size: 13px;
    }
    
    .footer-credit a {
        color: #00ff88;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)


class DividendDataFetcher:
    """Obtiene datos de dividendos desde dividendhistory.org"""
    
    def __init__(self):
        self.base_url = "https://dividendhistory.org/payout"
        self.session = requests.Session()
        self.cache = {}
    
    def fetch_dividends(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Obtiene dividendos mediante web scraping"""
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        url = f"{self.base_url}/{ticker}/"
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Connection': 'keep-alive',
            }
            
            response = self.session.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()
            
            if len(response.text) < 100:
                return pd.DataFrame()
            
            try:
                tables = pd.read_html(StringIO(response.text))
            except ImportError:
                return pd.DataFrame()
            
            if not tables:
                return pd.DataFrame()
            
            df = None
            for table in tables:
                temp_df = table.copy()
                temp_df.columns = [str(col).strip() for col in temp_df.columns]
                
                if all(col.isdigit() for col in temp_df.columns):
                    temp_df.columns = temp_df.iloc[0].astype(str).str.strip().tolist()
                    temp_df = temp_df.iloc[1:].reset_index(drop=True)
                
                if 'Ex-Dividend Date' in temp_df.columns or 'Cash Amount' in temp_df.columns:
                    df = temp_df
                    break
            
            if df is None:
                return pd.DataFrame()
            
            column_mapping = {
                'Ex-Dividend Date': 'ex_dividend_date',
                'Payout Date': 'payout_date',
                'Cash Amount': 'amount',
                '% Change': 'pct_change'
            }
            df = df.rename(columns=column_mapping)
            
            if 'ex_dividend_date' not in df.columns:
                return pd.DataFrame()
            
            df['ex_dividend_date'] = pd.to_datetime(df['ex_dividend_date'], errors='coerce')
            if 'payout_date' in df.columns:
                df['payout_date'] = pd.to_datetime(df['payout_date'], errors='coerce')
            
            if 'amount' in df.columns:
                df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            df = df[df['ex_dividend_date'].notna()]
            df = df[df['amount'].notna()]
            df = df[df['amount'] > 0]
            
            if start_date:
                df = df[df['ex_dividend_date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['ex_dividend_date'] <= pd.to_datetime(end_date)]
            
            df = df.sort_values('ex_dividend_date', ascending=False).reset_index(drop=True)
            
            if not df.empty:
                self.cache[cache_key] = df.copy()
            
            return df
            
        except Exception:
            return pd.DataFrame()


class GeraldineWeissAnalyzer:
    """Implementa el método de valoración por dividendos de Geraldine Weiss"""
    
    def __init__(self, ticker: str, years: int = 6):
        self.ticker = ticker
        self.years = years
        self.dividend_fetcher = DividendDataFetcher()
        self.data_source = None
        
    def fetch_price_data(self):
        """
        Obtiene datos históricos de precios sin caché usando Ticker().history()
        Esto evita problemas de caché de yf.download()
        """
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        try:
            # Usar Ticker().history() para evitar caché y obtener datos frescos
            ticker_obj = yf.Ticker(self.ticker)
            
            # history() sin auto_adjust para precios originales en moneda local
            data = ticker_obj.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
                actions=False
            )
            
            if data.empty:
                return None
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            data.index = pd.to_datetime(data.index)
            
            return data
            
        except Exception:
            return None
    
    def fetch_dividend_data(self):
        """Obtiene datos de dividendos priorizando dividendhistory.org para tickers USA"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        # Detectar si es ticker USA (sin sufijo)
        has_suffix = '.' in self.ticker
        
        # Para tickers USA, priorizar dividendhistory.org
        if not has_suffix:
            try:
                df = self.dividend_fetcher.fetch_dividends(
                    self.ticker, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if not df.empty:
                    self.data_source = "dividendhistory.org"
                    return df
            except Exception:
                pass
        
        # Si es internacional o falló dividendhistory, usar yfinance
        df = self._fetch_from_yfinance(start_date)
        if not df.empty:
            self.data_source = "yfinance"
            return df
        
        self.data_source = "none"
        return pd.DataFrame()
    
    def _fetch_from_yfinance(self, start_date):
        """Obtiene dividendos desde yfinance"""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            
            # Intentar con dividends
            divs = ticker_obj.dividends
            
            if divs.empty:
                # Intentar con actions
                try:
                    actions = ticker_obj.actions
                    if 'Dividends' in actions.columns:
                        divs = actions['Dividends']
                        divs = divs[divs > 0]
                except:
                    pass
            
            if not divs.empty:
                # Manejo de timezone
                if divs.index.tz is not None:
                    if start_date.tzinfo is None:
                        start_date = pytz.UTC.localize(start_date)
                    start_date = start_date.astimezone(divs.index.tz)
                else:
                    if start_date.tzinfo is not None:
                        start_date = start_date.replace(tzinfo=None)
                
                divs = divs[divs.index >= start_date]
                
                if not divs.empty:
                    df = pd.DataFrame({
                        'ex_dividend_date': divs.index.tz_localize(None) if divs.index.tz is not None else divs.index,
                        'amount': divs.values
                    })
                    df = df[df['amount'] > 0]
                    df = df.sort_values('ex_dividend_date', ascending=False).reset_index(drop=True)
                    return df
            
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()
    
    def calculate_annual_dividends(self, dividend_df):
        """Calcula dividendos anuales por año"""
        if dividend_df.empty:
            return pd.DataFrame()
        
        dividend_df = dividend_df.copy()
        dividend_df['year'] = dividend_df['ex_dividend_date'].dt.year
        
        # Filtrar outliers usando desviación estándar
        mean_div = dividend_df['amount'].mean()
        std_div = dividend_df['amount'].std()
        if std_div > 0:
            dividend_df = dividend_df[
                abs(dividend_df['amount'] - mean_div) <= (2.5 * std_div)
            ]
        
        annual = dividend_df.groupby('year')['amount'].sum().reset_index()
        annual.columns = ['year', 'annual_dividend']
        
        # Filtrar años con dividendos anormalmente bajos
        if len(annual) > 0:
            median_div = annual['annual_dividend'].median()
            annual = annual[annual['annual_dividend'] > median_div * 0.1]
        
        return annual
    
    def calculate_valuation_bands(self, prices, annual_dividends):
        """Calcula bandas de valoración según método Geraldine Weiss"""
        prices = prices.copy()
        
        if 'Close' not in prices.columns:
            if 'close' in prices.columns:
                prices = prices.rename(columns={'close': 'Close'})
            else:
                return None
        
        prices['year'] = prices.index.year
        
        merged = prices.merge(annual_dividends, on='year', how='inner')
        
        if merged.empty or len(merged) < 10:
            return None
        
        merged['div_yield'] = merged['annual_dividend'] / merged['Close']
        
        # Filtrar yields extremos
        yield_median = merged['div_yield'].median()
        yield_std = merged['div_yield'].std()
        if yield_std > 0:
            merged = merged[
                abs(merged['div_yield'] - yield_median) <= (3 * yield_std)
            ]
        
        if merged.empty:
            return None
        
        # Usar percentiles en vez de min/max para evitar outliers
        max_yield = merged['div_yield'].quantile(0.95)
        min_yield = merged['div_yield'].quantile(0.05)
        
        if max_yield == 0 or min_yield == 0 or max_yield <= min_yield:
            return None
        
        merged['undervalued'] = (merged['div_yield'] / max_yield) * merged['Close']
        merged['overvalued'] = (merged['div_yield'] / min_yield) * merged['Close']
        
        return merged
    
    def get_current_signal(self, analysis_df):
        """Determina la señal de inversión actual"""
        if analysis_df is None or analysis_df.empty:
            return "DESCONOCIDO", "Datos insuficientes", 0
        
        latest = analysis_df.iloc[-1]
        price = latest['Close']
        undervalued = latest['undervalued']
        overvalued = latest['overvalued']
        
        range_size = overvalued - undervalued
        lower_buy_zone = undervalued + (range_size * 0.2)
        upper_sell_zone = overvalued - (range_size * 0.2)
        
        if range_size > 0:
            score = ((overvalued - price) / range_size) * 200 - 100
        else:
            score = 0
        
        if price <= lower_buy_zone:
            return "COMPRA FUERTE", f"Precio {price:.2f} está en zona infravalorada", score
        elif price <= undervalued:
            return "COMPRA", f"Precio {price:.2f} se aproxima al nivel infravalorado", score
        elif price >= upper_sell_zone:
            return "VENTA FUERTE", f"Precio {price:.2f} está en zona sobrevalorada", score
        elif price >= overvalued:
            return "VENTA", f"Precio {price:.2f} se aproxima al nivel sobrevalorado", score
        else:
            return "MANTENER", f"Precio {price:.2f} está en rango de valor razonable", score


def analyze_ticker_quick(ticker, years=6):
    """Análisis rápido de un ticker"""
    try:
        analyzer = GeraldineWeissAnalyzer(ticker, years)
        
        price_data = analyzer.fetch_price_data()
        if price_data is None or price_data.empty:
            return None
        
        dividend_data = analyzer.fetch_dividend_data()
        if dividend_data.empty:
            return None
        
        annual_dividends = analyzer.calculate_annual_dividends(dividend_data)
        if annual_dividends.empty or len(annual_dividends) < 2:
            return None
        
        analysis_df = analyzer.calculate_valuation_bands(price_data, annual_dividends)
        if analysis_df is None or analysis_df.empty:
            return None
        
        signal, description, score = analyzer.get_current_signal(analysis_df)
        latest = analysis_df.iloc[-1]
        
        cagr = 0
        if len(annual_dividends) > 1:
            try:
                first_div = annual_dividends['annual_dividend'].iloc[0]
                last_div = annual_dividends['annual_dividend'].iloc[-1]
                if first_div > 0:
                    cagr = ((last_div / first_div) ** (1 / (len(annual_dividends) - 1)) - 1) * 100
            except:
                cagr = 0
        
        return {
            'ticker': ticker,
            'price': latest['Close'],
            'yield': latest['div_yield'] * 100,
            'undervalued': latest['undervalued'],
            'overvalued': latest['overvalued'],
            'signal': signal,
            'score': score,
            'annual_dividend': latest['annual_dividend'],
            'cagr': cagr,
            'analysis_df': analysis_df,
            'dividend_data': dividend_data,
            'data_source': analyzer.data_source
        }
    except Exception:
        return None


def create_weighted_portfolio_analysis(portfolio_results):
    """Crea análisis ponderado de cartera"""
    
    all_dates = set()
    for r in portfolio_results:
        all_dates.update(r['analysis_df'].index)
    
    all_dates = sorted(list(all_dates))
    
    portfolio_df = pd.DataFrame(index=all_dates)
    portfolio_df['weighted_price'] = 0.0
    portfolio_df['weighted_undervalued'] = 0.0
    portfolio_df['weighted_overvalued'] = 0.0
    
    for date in all_dates:
        total_weight_at_date = 0
        weighted_price = 0
        weighted_undervalued = 0
        weighted_overvalued = 0
        
        for r in portfolio_results:
            if date in r['analysis_df'].index:
                row = r['analysis_df'].loc[date]
            else:
                available_dates = r['analysis_df'].index[r['analysis_df'].index <= date]
                if len(available_dates) > 0:
                    closest_date = available_dates[-1]
                    row = r['analysis_df'].loc[closest_date]
                else:
                    continue
            
            weight = r['portfolio_weight'] / 100
            total_weight_at_date += weight
            
            weighted_price += row['Close'] * weight
            weighted_undervalued += row['undervalued'] * weight
            weighted_overvalued += row['overvalued'] * weight
        
        if total_weight_at_date > 0:
            portfolio_df.loc[date, 'weighted_price'] = weighted_price / total_weight_at_date
            portfolio_df.loc[date, 'weighted_undervalued'] = weighted_undervalued / total_weight_at_date
            portfolio_df.loc[date, 'weighted_overvalued'] = weighted_overvalued / total_weight_at_date
    
    portfolio_df = portfolio_df[(portfolio_df != 0).all(axis=1)]
    
    return portfolio_df


def get_data_source_badge(source):
    """Genera badge HTML para fuente de datos"""
    if source == "dividendhistory.org":
        return '<span class="data-source-badge source-dividendhistory">📊 dividendhistory.org</span>'
    elif source == "yfinance":
        return '<span class="data-source-badge source-yfinance">📈 yfinance</span>'
    else:
        return '<span class="data-source-badge">❓ Desconocido</span>'


def plot_geraldine_weiss_individual(analysis_df, ticker):
    """Gráfico de valoración Geraldine Weiss individual"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['overvalued'],
        name='Zona Sobrevalorada',
        line=dict(color='rgba(255, 107, 107, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['undervalued'],
        name='Rango de Valor Razonable',
        fill='tonexty',
        fillcolor='rgba(0, 255, 136, 0.1)',
        line=dict(color='rgba(0, 255, 136, 0)', width=0),
        showlegend=True,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['overvalued'],
        name='Sobrevalorada',
        line=dict(color='#ff6b6b', width=3),
        mode='lines',
        hovertemplate='<b>Sobrevalorada:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['undervalued'],
        name='Infravalorada',
        line=dict(color='#00ff88', width=3),
        mode='lines',
        hovertemplate='<b>Infravalorada:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['Close'],
        name='Precio Actual',
        line=dict(color='#00d4ff', width=4),
        mode='lines',
        hovertemplate='<b>Precio:</b> %{y:.2f}<extra></extra>'
    ))
    
    latest = analysis_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[analysis_df.index[-1]],
        y=[latest['Close']],
        mode='markers',
        marker=dict(size=16, color='#00d4ff', line=dict(color='white', width=3)),
        showlegend=False,
        hovertemplate=f'<b>Actual: {latest["Close"]:.2f}</b><extra></extra>'
    ))
    
    fig.add_annotation(
        x=analysis_df.index[-1],
        y=latest['Close'],
        text=f"{latest['Close']:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#00d4ff",
        ax=40,
        ay=-40,
        bgcolor="rgba(0, 212, 255, 0.2)",
        bordercolor="#00d4ff",
        borderwidth=2,
        font=dict(size=14, color="white")
    )
    
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> - Modelo de Valoración Geraldine Weiss',
            font=dict(size=24, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False,
            type='date',  # ✅ AÑADIDO
            tickformat='%b %Y',  # ✅ AÑADIDO - Formato: Ene 2024
            dtick='M6'  # ✅ AÑADIDO - Tick cada 6 meses
        ),
        yaxis=dict(
            title='Precio',
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False
        ),
        template='plotly_dark',
        hovermode='x unified',
        height=550,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 40, 57, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        )
    )
    
    return fig


def plot_portfolio_geraldine_weiss(portfolio_df):
    """Gráfico de valoración de cartera ponderada"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['weighted_overvalued'],
        name='Zona Sobrevalorada',
        line=dict(color='rgba(255, 107, 107, 0)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['weighted_undervalued'],
        name='Rango de Valor Razonable',
        fill='tonexty',
        fillcolor='rgba(0, 255, 136, 0.1)',
        line=dict(color='rgba(0, 255, 136, 0)', width=0),
        showlegend=True,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['weighted_overvalued'],
        name='Sobrevalorada',
        line=dict(color='#ff6b6b', width=3),
        mode='lines',
        hovertemplate='<b>Sobrevalorada:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['weighted_undervalued'],
        name='Infravalorada',
        line=dict(color='#00ff88', width=3),
        mode='lines',
        hovertemplate='<b>Infravalorada:</b> %{y:.2f}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=portfolio_df.index,
        y=portfolio_df['weighted_price'],
        name='Precio Ponderado',
        line=dict(color='#00d4ff', width=4),
        mode='lines',
        hovertemplate='<b>Precio:</b> %{y:.2f}<extra></extra>'
    ))
    
    latest = portfolio_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[portfolio_df.index[-1]],
        y=[latest['weighted_price']],
        mode='markers',
        marker=dict(size=16, color='#00d4ff', line=dict(color='white', width=3)),
        showlegend=False,
        hovertemplate=f'<b>Actual: {latest["weighted_price"]:.2f}</b><extra></extra>'
    ))
    
    fig.add_annotation(
        x=portfolio_df.index[-1],
        y=latest['weighted_price'],
        text=f"{latest['weighted_price']:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#00d4ff",
        ax=40,
        ay=-40,
        bgcolor="rgba(0, 212, 255, 0.2)",
        bordercolor="#00d4ff",
        borderwidth=2,
        font=dict(size=14, color="white")
    )
    
    fig.update_layout(
        title=dict(
            text='<b>Cartera Ponderada</b> - Análisis Geraldine Weiss',
            font=dict(size=24, color='white'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False,
            type='date',  # ✅ AÑADIDO
            tickformat='%b %Y',  # ✅ AÑADIDO - Formato: Ene 2024
            dtick='M6'  # ✅ AÑADIDO - Tick cada 6 meses
        ),
        yaxis=dict(
            title='Valor Ponderado',
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False
        ),
        template='plotly_dark',
        hovermode='x unified',
        height=550,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(30, 40, 57, 0.8)',
            bordercolor='rgba(255, 255, 255, 0.2)',
            borderwidth=1
        )
    )
    
    return fig


def plot_comparison_chart(results):
    """Gráfico comparativo de múltiples tickers"""
    fig = go.Figure()
    
    tickers = [r['ticker'] for r in results]
    prices = [r['price'] for r in results]
    undervalued = [r['undervalued'] for r in results]
    overvalued = [r['overvalued'] for r in results]
    
    x = list(range(len(tickers)))
    
    fig.add_trace(go.Scatter(
        x=x, y=overvalued,
        name='Sobrevalorada',
        line=dict(color='#ff6b6b', width=2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=undervalued,
        name='Infravalorada',
        line=dict(color='#00ff88', width=2, dash='dash'),
        mode='lines+markers',
        marker=dict(size=10)
    ))
    
    colors = []
    for r in results:
        if 'COMPRA' in r['signal']:
            colors.append('#00ff88')
        elif 'VENTA' in r['signal']:
            colors.append('#ff6b6b')
        else:
            colors.append('#ffd93d')
    
    fig.add_trace(go.Scatter(
        x=x, y=prices,
        name='Precio Actual',
        mode='lines+markers',
        line=dict(color='#00d4ff', width=4),
        marker=dict(size=15, color=colors, line=dict(color='white', width=2)),
        text=[f"{p:.2f}" for p in prices],
        textposition="top center"
    ))
    
    fig.update_layout(
        title='Comparación de Valoración - Método Geraldine Weiss',
        xaxis=dict(
            tickmode='array',
            tickvals=x,
            ticktext=tickers,
            title=''
        ),
        yaxis=dict(
            title='Precio'
        ),
        template='plotly_dark',
        height=500,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


def plot_portfolio_composition(portfolio_data):
    """Gráfico de composición de cartera"""
    fig = go.Figure()
    
    labels = [f"{row['ticker']}<br>{row['weight']:.1f}%" for _, row in portfolio_data.iterrows()]
    values = portfolio_data['weight'].tolist()
    
    colors = ['#00ff88', '#00d4ff', '#7b2ff7', '#ffd93d', '#ff6b6b', 
              '#51cf66', '#ff8787', '#00b4d8', '#90e0ef', '#ff006e']
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors[:len(labels)], line=dict(color='#1a1f2e', width=2)),
        textinfo='label+percent',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>Peso: %{value:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Composición de la Cartera',
        template='plotly_dark',
        height=450,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        showlegend=False
    )
    
    return fig


def main():
    st.title("💎 Geraldine Weiss - Análisis de Dividendos")
    st.caption("Plataforma Profesional de Valoración por Dividendos y Estrategia de Inversión")
    
    main_tab1, main_tab2, main_tab3 = st.tabs([
        "🎯 Análisis Individual",
        "📊 Comparación Multi-Ticker",
        "💼 Cartera Ponderada"
    ])
    
    # ==================== TAB 1: ANÁLISIS INDIVIDUAL ====================
    with main_tab1:
        col_sidebar, col_main = st.columns([1, 3])
        
        with col_sidebar:
            st.header("⚙️ Configuración")
            
            ticker = st.text_input(
                "Stock Ticker",
                value="KO",
                help="Introduce el símbolo de una acción que pague dividendos"
            )
            
            years = st.slider(
                "Período de Análisis (años)",
                min_value=3,
                max_value=10,
                value=6
            )
            
            st.divider()
            
            analyze_button = st.button(
                "🔍 Analizar Acción",
                type="primary",
                use_container_width=True
            )
            
            st.divider()
            
            with st.expander("💡 Sobre Este Método"):
                st.markdown("""
                El enfoque de **Geraldine Weiss** identifica valor mediante análisis de rentabilidad por dividendo:
                
                - **Alta rentabilidad** = Infravalorada (Compra)
                - **Baja rentabilidad** = Sobrevalorada (Venta)
                - **Rango medio** = Valor razonable (Mantener)
                """)
            
            with st.expander("📌 Sobre Geraldine Weiss"):
                st.markdown("""
                Geraldine Weiss fue pionera en la teoría de valoración mediante rentabilidad por dividendo.
                
                **Este método funciona mejor con:**
                - ✓ Aristócratas de Dividendos
                - ✓ Pagadores estables de dividendos
                - ✓ Acciones blue-chip
                """)
            
            st.divider()
            
            st.markdown("**🎯 Tickers Sugeridos**")
            st.caption("**USA:** KO · JNJ · PG · MMM · CAT · XOM")
            st.caption("**Europa:** IBE.MC · SAN.MC · TEF.MC · VOW3.DE")
            st.caption("**UK:** ULVR.L · BP.L · HSBA.L")
            st.caption("**Canadá:** RY.TO · TD.TO · ENB.TO")
        
        with col_main:
            if analyze_button and ticker:
                with st.spinner('🔄 Analizando datos del mercado...'):
                    result = analyze_ticker_quick(ticker.upper(), years)
                    
                    if result is None:
                        st.error(f"""
                        ❌ **No se pudieron obtener datos suficientes para {ticker.upper()}**
                        
                        **Posibles causas:**
                        - El ticker no existe o está mal escrito
                        - La acción no paga dividendos o tiene historial muy limitado
                        - No hay suficiente historial de datos ({years} años)
                        
                        💡 **Sugerencias:**
                        - Verifica que el ticker sea correcto
                        - Para mercados internacionales usa el sufijo correcto:
                          - España: .MC (IBE.MC)
                          - UK: .L (BP.L)
                          - Alemania: .DE (VOW3.DE)
                          - Canadá: .TO (RY.TO)
                        - Reduce el período de análisis a 3 años
                        - Asegúrate de que la empresa pague dividendos regularmente
                        """)
                    else:
                        st.markdown(
                            f"✅ **Análisis completado para {ticker.upper()}** {get_data_source_badge(result['data_source'])}",
                            unsafe_allow_html=True
                        )
                        
                        signal_colors = {
                            "COMPRA FUERTE": "#00ff88",
                            "COMPRA": "#51cf66",
                            "MANTENER": "#ffd93d",
                            "VENTA": "#ff8787",
                            "VENTA FUERTE": "#ff6b6b"
                        }
                        
                        st.markdown(
                            f"""<div class='big-signal' style='border-color: {signal_colors.get(result['signal'], "#ffffff")}; color: {signal_colors.get(result['signal'], "#ffffff")}'>
                            {result['signal']}<br>
                            <div style='font-size: 18px; margin-top: 10px;'>Precio: {result['price']:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        st.subheader("📊 Métricas Clave")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        upside_undervalued = ((result['undervalued']/result['price'] - 1) * 100)
                        upside_overvalued = ((result['overvalued']/result['price'] - 1) * 100)
                        
                        col1.metric("💵 Precio Actual", f"{result['price']:.2f}")
                        col2.metric("📊 Rentabilidad", f"{result['yield']:.2f}%")
                        col3.metric("🟢 Zona Infravalorada", f"{result['undervalued']:.2f}", 
                                   delta=f"{upside_undervalued:.1f}%", delta_color="inverse")
                        col4.metric("🔴 Zona Sobrevalorada", f"{result['overvalued']:.2f}",
                                   delta=f"{upside_overvalued:.1f}%")
                        
                        st.divider()
                        
                        st.subheader("📈 Análisis de Valoración")
                        st.plotly_chart(
                            plot_geraldine_weiss_individual(result['analysis_df'], ticker.upper()),
                            use_container_width=True
                        )
                        
                        st.divider()
                        
                        st.subheader("🎯 Interpretación")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.info("""
                            **🟢 Zona Infravalorada (Verde)**
                            
                            Cuando el precio se aproxima a la línea verde, la acción ofrece alta rentabilidad por dividendo.
                            
                            - **Acción:** Considerar comprar
                            - **Riesgo:** Menor relativo al histórico
                            - **Retorno:** Apreciación + dividendos
                            """)
                        
                        with col2:
                            st.warning("""
                            **🔴 Zona Sobrevalorada (Roja)**
                            
                            Cuando el precio se aproxima a la línea roja, la acción ofrece baja rentabilidad por dividendo.
                            
                            - **Acción:** Considerar vender
                            - **Riesgo:** Mayor relativo al histórico
                            - **Retorno:** Potencial alcista limitado
                            """)
            else:
                st.info("""
                ### 👋 Bienvenido al Analizador Geraldine Weiss
                
                Introduce un ticker en la barra lateral y haz clic en **"Analizar Acción"** para comenzar.
                
                **Ejemplo de tickers:**
                - **USA:** KO, JNJ, PG, MMM
                - **Europa:** IBE.MC, SAN.MC, VOW3.DE
                - **UK:** BP.L, ULVR.L, HSBA.L
                - **Canadá:** RY.TO, TD.TO, ENB.TO
                """)
    
    # ==================== TAB 2: COMPARACIÓN MULTI-TICKER ====================
    with main_tab2:
        st.header("📊 Análisis Comparativo Multi-Ticker")
        st.caption("Compara hasta 6 acciones simultáneamente usando el método Geraldine Weiss")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            tickers_input = st.text_input(
                "Tickers a Comparar (separados por comas)",
                value="KO, PG, JNJ, PEP",
                help="Ejemplo: KO, PG, JNJ, PEP"
            )
        
        with col2:
            years_comp = st.selectbox("Período", [3, 5, 6, 10], index=2)
            compare_button = st.button("🔍 Comparar Acciones", type="primary", use_container_width=True)
        
        if compare_button:
            tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            if len(tickers_list) < 2:
                st.error("❌ Por favor introduce al menos 2 tickers para comparar")
            elif len(tickers_list) > 6:
                st.warning("⚠️ Máximo 6 tickers. Se analizarán los primeros 6.")
                tickers_list = tickers_list[:6]
            else:
                with st.spinner(f'🔄 Analizando {len(tickers_list)} acciones...'):
                    results = []
                    failed_tickers = []
                    progress_bar = st.progress(0)
                    
                    for i, ticker in enumerate(tickers_list):
                        result = analyze_ticker_quick(ticker, years_comp)
                        if result:
                            results.append(result)
                        else:
                            failed_tickers.append(ticker)
                        progress_bar.progress((i + 1) / len(tickers_list))
                    
                    progress_bar.empty()
                    
                    if failed_tickers:
                        st.warning(f"⚠️ No se pudieron analizar: {', '.join(failed_tickers)}")
                    
                    if not results:
                        st.error("❌ **No se pudieron obtener datos para ningún ticker**")
                    else:
                        st.success(f"✅ Análisis completado para {len(results)} acciones")
                        
                        st.plotly_chart(plot_comparison_chart(results), use_container_width=True)
                        
                        st.divider()
                        
                        st.subheader("📋 Tabla Comparativa")
                        
                        comparison_df = pd.DataFrame([{
                            'Ticker': r['ticker'],
                            'Fuente': '📊' if r['data_source'] == 'dividendhistory.org' else '📈',
                            'Precio': f"{r['price']:.2f}",
                            'Yield': f"{r['yield']:.2f}%",
                            'Infravalorada': f"{r['undervalued']:.2f}",
                            'Sobrevalorada': f"{r['overvalued']:.2f}",
                            'Señal': r['signal'],
                            'Score': f"{r['score']:.1f}",
                            'CAGR Div.': f"{r['cagr']:.1f}%"
                        } for r in results])
                        
                        comparison_df = comparison_df.sort_values('Score', ascending=False)
                        
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        st.caption("📊 = dividendhistory.org | 📈 = yfinance")
                        
                        st.divider()
                        st.subheader("🏆 Ranking de Oportunidades")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        buy_opps = sorted([r for r in results if 'COMPRA' in r['signal']], 
                                        key=lambda x: x['score'], reverse=True)
                        hold_opps = [r for r in results if r['signal'] == 'MANTENER']
                        sell_opps = sorted([r for r in results if 'VENTA' in r['signal']], 
                                         key=lambda x: x['score'])
                        
                        with col1:
                            st.success("**🟢 Mejores Oportunidades de Compra**")
                            if buy_opps:
                                for r in buy_opps[:3]:
                                    st.markdown(f"**{r['ticker']}** - {r['signal']}")
                                    st.caption(f"Yield: {r['yield']:.2f}% | Score: {r['score']:.1f}")
                            else:
                                st.caption("No hay señales de compra")
                        
                        with col2:
                            st.info("**🟡 Mantener Posición**")
                            if hold_opps:
                                for r in hold_opps[:3]:
                                    st.markdown(f"**{r['ticker']}** - {r['signal']}")
                                    st.caption(f"Yield: {r['yield']:.2f}%")
                            else:
                                st.caption("No hay señales de mantener")
                        
                        with col3:
                            st.warning("**🔴 Considerar Venta**")
                            if sell_opps:
                                for r in sell_opps[:3]:
                                    st.markdown(f"**{r['ticker']}** - {r['signal']}")
                                    st.caption(f"Yield: {r['yield']:.2f}% | Score: {r['score']:.1f}")
                            else:
                                st.caption("No hay señales de venta")
    
    # ==================== TAB 3: CARTERA PONDERADA ====================
    with main_tab3:
        st.header("💼 Análisis de Cartera Ponderada")
        st.caption("Aplica el método Geraldine Weiss a tu cartera completa")
        
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'weight'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("➕ Añadir Posiciones")
            
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                new_ticker = st.text_input("Ticker", key="portfolio_ticker")
            with col_b:
                new_weight = st.number_input("Peso (%)", min_value=0.0, max_value=100.0, 
                                            value=10.0, step=5.0, key="portfolio_weight")
            with col_c:
                st.write("")
                st.write("")
                if st.button("➕ Añadir", type="secondary", use_container_width=True):
                    if new_ticker:
                        new_row = pd.DataFrame([{'ticker': new_ticker.upper(), 'weight': new_weight}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], 
                                                              ignore_index=True)
                        st.rerun()
        
        with col2:
            st.subheader("🎯 Acciones Rápidas")
            if st.button("🗑️ Limpiar Cartera", use_container_width=True):
                st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'weight'])
                st.rerun()
            
            if st.button("📋 Cartera Ejemplo", use_container_width=True):
                st.session_state.portfolio = pd.DataFrame([
                    {'ticker': 'KO', 'weight': 25},
                    {'ticker': 'JNJ', 'weight': 25},
                    {'ticker': 'PG', 'weight': 25},
                    {'ticker': 'PEP', 'weight': 25}
                ])
                st.rerun()
        
        st.divider()
        
        if not st.session_state.portfolio.empty:
            st.subheader("📊 Cartera Actual")
            
            total_weight = st.session_state.portfolio['weight'].sum()
            if total_weight != 100:
                st.warning(f"⚠️ Los pesos suman {total_weight:.1f}%. Se normalizarán a 100%.")
            
            edited_df = st.data_editor(
                st.session_state.portfolio,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic"
            )
            
            st.session_state.portfolio = edited_df
            
            st.divider()
            
            if st.button("🔍 Analizar Cartera Completa", type="primary", use_container_width=True):
                with st.spinner('🔄 Analizando cartera...'):
                    portfolio_data = st.session_state.portfolio.copy()
                    portfolio_data['weight'] = (portfolio_data['weight'] / portfolio_data['weight'].sum()) * 100
                    
                    portfolio_results = []
                    failed_tickers = []
                    progress_bar = st.progress(0)
                    
                    for i, row in portfolio_data.iterrows():
                        result = analyze_ticker_quick(row['ticker'], 6)
                        if result:
                            result['portfolio_weight'] = row['weight']
                            portfolio_results.append(result)
                        else:
                            failed_tickers.append(row['ticker'])
                        progress_bar.progress((i + 1) / len(portfolio_data))
                    
                    progress_bar.empty()
                    
                    if failed_tickers:
                        st.warning(f"⚠️ No se pudieron analizar: {', '.join(failed_tickers)}")
                    
                    if not portfolio_results:
                        st.error("❌ No se pudieron obtener datos para ningún ticker de la cartera")
                    else:
                        st.success(f"✅ Cartera analizada: {len(portfolio_results)} posiciones")
                        
                        total_yield = sum(r['yield'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        total_cagr = sum(r['cagr'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        avg_score = sum(r['score'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        
                        if avg_score > 30:
                            portfolio_signal = "COMPRA"
                            signal_color = "#00ff88"
                        elif avg_score < -30:
                            portfolio_signal = "VENTA"
                            signal_color = "#ff6b6b"
                        else:
                            portfolio_signal = "MANTENER"
                            signal_color = "#ffd93d"
                        
                        st.markdown(
                            f"""<div class='big-signal' style='border-color: {signal_color}; color: {signal_color}; font-size: 36px;'>
                            Señal de Cartera: {portfolio_signal}<br>
                            <div style='font-size: 16px; margin-top: 10px;'>Score Ponderado: {avg_score:.1f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        st.subheader("📊 Métricas de Cartera Ponderada")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("💰 Yield Promedio", f"{total_yield:.2f}%")
                        col2.metric("📈 CAGR Promedio", f"{total_cagr:.2f}%")
                        col3.metric("🎯 Score Cartera", f"{avg_score:.1f}")
                        col4.metric("📋 Posiciones", len(portfolio_results))
                        
                        st.divider()
                        
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.plotly_chart(plot_portfolio_composition(portfolio_data), 
                                          use_container_width=True)
                        
                        with col2:
                            st.subheader("📋 Detalle de Posiciones")
                            
                            portfolio_detail = pd.DataFrame([{
                                'Ticker': r['ticker'],
                                'Fuente': '📊' if r['data_source'] == 'dividendhistory.org' else '📈',
                                'Peso': f"{r['portfolio_weight']:.1f}%",
                                'Yield': f"{r['yield']:.2f}%",
                                'Señal': r['signal'],
                                'Score': f"{r['score']:.1f}"
                            } for r in portfolio_results])
                            
                            st.dataframe(portfolio_detail, use_container_width=True, hide_index=True)
                            st.caption("📊 = dividendhistory.org | 📈 = yfinance")
                        
                        st.divider()
                        
                        st.subheader("📈 Análisis Geraldine Weiss de Cartera Ponderada")
                        st.caption("Este gráfico muestra la valoración agregada de toda la cartera, ponderando cada posición según su peso")
                        
                        portfolio_df = create_weighted_portfolio_analysis(portfolio_results)
                        st.plotly_chart(
                            plot_portfolio_geraldine_weiss(portfolio_df),
                            use_container_width=True
                        )
                        
                        st.divider()
                        
                        st.subheader("💡 Recomendaciones de Rebalanceo")
                        
                        buy_pos = [r for r in portfolio_results if 'COMPRA' in r['signal']]
                        sell_pos = [r for r in portfolio_results if 'VENTA' in r['signal']]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success("**🟢 Considerar Aumentar Posición**")
                            if buy_pos:
                                for r in buy_pos:
                                    st.markdown(f"**{r['ticker']}** ({r['portfolio_weight']:.1f}%) - {r['signal']}")
                                    st.caption(f"Yield: {r['yield']:.2f}% | Score: {r['score']:.1f}")
                            else:
                                st.caption("No hay posiciones en zona de compra")
                        
                        with col2:
                            st.warning("**🔴 Considerar Reducir Posición**")
                            if sell_pos:
                                for r in sell_pos:
                                    st.markdown(f"**{r['ticker']}** ({r['portfolio_weight']:.1f}%) - {r['signal']}")
                                    st.caption(f"Yield: {r['yield']:.2f}% | Score: {r['score']:.1f}")
                            else:
                                st.caption("No hay posiciones en zona de venta")
        else:
            st.info("""
            ### 📝 Instrucciones
            
            1. **Añade tickers** con sus pesos porcentuales
            2. Los pesos se **normalizarán automáticamente** a 100%
            3. Haz clic en **"Analizar Cartera Completa"**
            4. Obtén señales y recomendaciones ponderadas
            
            💡 **Tip:** Usa el botón "Cartera Ejemplo" para ver un ejemplo rápido
            """)
    
    st.markdown("""
    <div class='footer-credit'>
        Desarrollado por <a href='https://bquantfinance.com' target='_blank'>@Gsnchez | bquantfinance.com</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
