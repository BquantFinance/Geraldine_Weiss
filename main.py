import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import requests
from io import StringIO

# Configuración de la página
st.set_page_config(
    page_title="Geraldine Weiss | Estrategia de Dividendos",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Tema oscuro premium con CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        background-attachment: fixed;
    }
    
    .stApp {
        background: transparent;
    }
    
    /* Tarjetas de cristal (glass morphism) */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 30px;
        margin: 20px 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px 0 rgba(0, 255, 136, 0.15);
        border: 1px solid rgba(0, 255, 136, 0.2);
    }
    
    /* Tarjetas de métricas premium */
    .metric-container {
        background: linear-gradient(135deg, rgba(10, 14, 39, 0.9) 0%, rgba(26, 31, 58, 0.9) 100%);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .metric-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #00ff88, #00d4ff, #7b2ff7);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    
    .metric-container:hover::before {
        opacity: 1;
    }
    
    .metric-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 48px rgba(0, 255, 136, 0.25);
        border: 1px solid rgba(0, 255, 136, 0.3);
    }
    
    /* Banner de señal */
    .signal-banner {
        background: linear-gradient(135deg, rgba(10, 14, 39, 0.95) 0%, rgba(26, 31, 58, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 48px;
        text-align: center;
        border: 2px solid;
        margin: 30px 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
    }
    
    .signal-banner::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(0, 255, 136, 0.1) 0%, transparent 70%);
        animation: pulse 3s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .signal-title {
        font-size: 48px;
        font-weight: 700;
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 0 0 30px currentColor;
        position: relative;
        z-index: 1;
    }
    
    .signal-description {
        font-size: 20px;
        margin: 16px 0 0 0;
        font-weight: 400;
        opacity: 0.9;
        position: relative;
        z-index: 1;
    }
    
    /* Valores de métricas */
    div[data-testid="stMetricValue"] {
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: rgba(255, 255, 255, 0.6);
        font-weight: 600;
    }
    
    div[data-testid="stMetricDelta"] {
        font-size: 16px;
        font-weight: 600;
    }
    
    /* Encabezados */
    h1 {
        font-size: 56px !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #ffffff 0%, #00ff88 50%, #00d4ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 8px !important;
        letter-spacing: -1px;
    }
    
    h2 {
        font-size: 32px !important;
        font-weight: 600 !important;
        color: #ffffff !important;
        margin-top: 40px !important;
        margin-bottom: 20px !important;
    }
    
    h3 {
        font-size: 24px !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.95) !important;
        margin-top: 32px !important;
        margin-bottom: 16px !important;
    }
    
    /* Subtítulo */
    .subtitle {
        font-size: 20px;
        color: rgba(255, 255, 255, 0.7);
        font-weight: 400;
        margin-top: -10px;
        margin-bottom: 40px;
    }
    
    /* Pestañas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(255, 255, 255, 0.02);
        padding: 8px;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 12px;
        padding: 16px 32px;
        font-weight: 600;
        font-size: 15px;
        color: rgba(255, 255, 255, 0.6);
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.05);
        color: rgba(255, 255, 255, 0.9);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 212, 255, 0.15) 100%) !important;
        border: 1px solid rgba(0, 255, 136, 0.3) !important;
        color: #00ff88 !important;
    }
    
    /* Barra lateral */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 14, 39, 0.95) 0%, rgba(26, 31, 58, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Campos de entrada */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        color: #ffffff;
        font-size: 16px;
        padding: 12px 16px;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border: 1px solid rgba(0, 255, 136, 0.5);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.2);
        background: rgba(255, 255, 255, 0.08);
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        color: #0a0e27;
        border: none;
        border-radius: 12px;
        padding: 16px 32px;
        font-weight: 700;
        font-size: 16px;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 8px 24px rgba(0, 255, 136, 0.3);
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 32px rgba(0, 255, 136, 0.5);
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
    }
    
    /* Deslizador */
    .stSlider > div > div > div {
        background: rgba(0, 255, 136, 0.2);
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    /* Cajas de información */
    .info-box {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.08) 0%, rgba(123, 47, 247, 0.08) 100%);
        border-left: 4px solid #00d4ff;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.08) 0%, rgba(0, 212, 255, 0.08) 100%);
        border-left: 4px solid #00ff88;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 136, 0.2);
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 217, 61, 0.08) 0%, rgba(255, 107, 107, 0.08) 100%);
        border-left: 4px solid #ffd93d;
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 217, 61, 0.2);
    }
    
    /* Pantalla de bienvenida */
    .welcome-container {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.05) 0%, rgba(0, 212, 255, 0.05) 100%);
        border-radius: 24px;
        padding: 60px;
        margin: 40px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
        text-align: center;
    }
    
    /* Estilo de dataframe */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Spinner de carga */
    .stSpinner > div {
        border-top-color: #00ff88 !important;
    }
    
    /* Barra de desplazamiento */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #00ff88 0%, #00d4ff 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 100%);
    }
    
    /* Divisor */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        margin: 40px 0;
    }
    
    /* Eliminar marca de Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Texto responsivo */
    @media (max-width: 768px) {
        h1 { font-size: 36px !important; }
        .signal-title { font-size: 32px; }
        .signal-description { font-size: 16px; }
    }
    
    /* Créditos del autor */
    .author-credit {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, rgba(10, 14, 39, 0.95) 0%, rgba(26, 31, 58, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 12px;
        padding: 12px 20px;
        border: 1px solid rgba(0, 255, 136, 0.3);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        z-index: 999;
        transition: all 0.3s ease;
    }
    
    .author-credit:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 255, 136, 0.3);
    }
    
    .author-credit a {
        color: #00ff88;
        text-decoration: none;
        font-weight: 600;
        font-size: 14px;
    }
    
    .author-credit a:hover {
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)


class DividendDataFetcher:
    """Obtiene datos de dividendos desde dividendhistory.org"""
    
    def __init__(self):
        self.base_url = "https://dividendhistory.org/payout"
        self.session = requests.Session()
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}
    
    def fetch_dividends(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Obtiene dividendos mediante web scraping"""
        cache_key = f"{ticker}_{start_date}_{end_date}"
        if cache_key in self.cache:
            return self.cache[cache_key].copy()
        
        url = f"{self.base_url}/{ticker}/"
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            tables = pd.read_html(StringIO(response.text))
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
                df['amount'] = df['amount'].astype(str).str.replace('$', '').str.replace(',', '').str.strip()
                df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
            
            df = df[df['ex_dividend_date'].notna()]
            
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
        
    def fetch_price_data(self):
        """Obtiene datos históricos de precios"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                return None
            data.index = pd.to_datetime(data.index)
            return data
        except Exception:
            return None
    
    def fetch_dividend_data(self):
        """Obtiene datos de dividendos"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        df = self.dividend_fetcher.fetch_dividends(
            self.ticker, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            try:
                ticker_obj = yf.Ticker(self.ticker)
                divs = ticker_obj.dividends
                if not divs.empty:
                    df = pd.DataFrame({
                        'ex_dividend_date': divs.index,
                        'amount': divs.values
                    })
                    df = df[df['ex_dividend_date'] >= start_date]
            except:
                pass
        
        return df
    
    def calculate_annual_dividends(self, dividend_df):
        """Calcula dividendos anuales por año"""
        if dividend_df.empty:
            return pd.DataFrame()
        
        dividend_df = dividend_df.copy()
        dividend_df['year'] = dividend_df['ex_dividend_date'].dt.year
        
        mean_div = dividend_df['amount'].mean()
        std_div = dividend_df['amount'].std()
        dividend_df = dividend_df[
            abs(dividend_df['amount'] - mean_div) <= (2.5 * std_div)
        ]
        
        annual = dividend_df.groupby('year')['amount'].sum().reset_index()
        annual.columns = ['year', 'annual_dividend']
        
        return annual
    
    def calculate_valuation_bands(self, prices, annual_dividends):
        """Calcula bandas de sobrevaloración e infravaloración"""
        prices = prices.copy()
        prices['year'] = prices.index.year
        
        merged = prices.merge(annual_dividends, on='year', how='inner')
        
        if merged.empty:
            return None
        
        merged['div_yield'] = merged['annual_dividend'] / merged['Close']
        
        max_yield = merged['div_yield'].max()
        min_yield = merged['div_yield'].min()
        
        if max_yield == 0 or min_yield == 0:
            return None
        
        merged['undervalued'] = (merged['div_yield'] / max_yield) * merged['Close']
        merged['overvalued'] = (merged['div_yield'] / min_yield) * merged['Close']
        
        return merged
    
    def get_current_signal(self, analysis_df):
        """Determina la señal actual de compra/venta"""
        if analysis_df is None or analysis_df.empty:
            return "DESCONOCIDO", "Datos insuficientes"
        
        latest = analysis_df.iloc[-1]
        price = latest['Close']
        undervalued = latest['undervalued']
        overvalued = latest['overvalued']
        
        range_size = overvalued - undervalued
        lower_buy_zone = undervalued + (range_size * 0.2)
        upper_sell_zone = overvalued - (range_size * 0.2)
        
        if price <= lower_buy_zone:
            return "COMPRA FUERTE", f"Precio ${price:.2f} está en zona infravalorada"
        elif price <= undervalued:
            return "COMPRA", f"Precio ${price:.2f} se aproxima al nivel infravalorado"
        elif price >= upper_sell_zone:
            return "VENTA FUERTE", f"Precio ${price:.2f} está en zona sobrevalorada"
        elif price >= overvalued:
            return "VENTA", f"Precio ${price:.2f} se aproxima al nivel sobrevalorado"
        else:
            return "MANTENER", f"Precio ${price:.2f} está en rango de valor razonable"


def plot_geraldine_weiss(analysis_df, ticker):
    """Crea gráfico premium de Geraldine Weiss"""
    fig = go.Figure()
    
    # Área de relleno entre bandas
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['overvalued'],
        name='Sobrevalorada',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['undervalued'],
        name='Rango de Valor Razonable',
        fill='tonexty',
        fillcolor='rgba(0, 255, 136, 0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        showlegend=True,
        hoverinfo='skip'
    ))
    
    # Línea de sobrevaloración
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['overvalued'],
        name='Zona Sobrevalorada',
        line=dict(color='#ff6b6b', width=3, dash='solid'),
        mode='lines',
        hovertemplate='<b>Sobrevalorada</b><br>Precio: $%{y:.2f}<br>Fecha: %{x}<extra></extra>'
    ))
    
    # Línea de infravaloración
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['undervalued'],
        name='Zona Infravalorada',
        line=dict(color='#00ff88', width=3, dash='solid'),
        mode='lines',
        hovertemplate='<b>Infravalorada</b><br>Precio: $%{y:.2f}<br>Fecha: %{x}<extra></extra>'
    ))
    
    # Precio actual con efecto de degradado
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['Close'],
        name='Precio Actual',
        line=dict(
            color='#00d4ff',
            width=4,
        ),
        mode='lines',
        hovertemplate='<b>Precio</b><br>$%{y:.2f}<br>Fecha: %{x}<extra></extra>'
    ))
    
    # Marcador para el precio actual
    latest = analysis_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[analysis_df.index[-1]],
        y=[latest['Close']],
        mode='markers',
        marker=dict(
            size=15,
            color='#00d4ff',
            line=dict(color='white', width=2)
        ),
        showlegend=False,
        hovertemplate=f'<b>Actual: ${latest["Close"]:.2f}</b><extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> · Modelo de Valoración Geraldine Weiss',
            font=dict(size=28, color='white', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=True,
            zeroline=False,
            color='rgba(255, 255, 255, 0.7)'
        ),
        yaxis=dict(
            title='Precio (USD)',
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=True,
            zeroline=False,
            color='rgba(255, 255, 255, 0.7)',
            tickprefix='$'
        ),
        plot_bgcolor='rgba(10, 14, 39, 0.5)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        hovermode='x unified',
        height=550,
        font=dict(family='Inter', color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(255, 255, 255, 0.05)',
            bordercolor='rgba(255, 255, 255, 0.1)',
            borderwidth=1,
            font=dict(size=12)
        ),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


def plot_dividend_history(dividend_df, ticker):
    """Crea gráfico premium del historial de dividendos"""
    if dividend_df.empty:
        return None
    
    fig = go.Figure()
    
    colors = ['#00ff88' if i % 2 == 0 else '#00d4ff' for i in range(len(dividend_df))]
    
    fig.add_trace(go.Bar(
        x=dividend_df['ex_dividend_date'],
        y=dividend_df['amount'],
        name='Dividendo',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
        ),
        hovertemplate='<b>%{x|%Y-%m-%d}</b><br>Monto: $%{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f'<b>{ticker}</b> · Historial de Pagos de Dividendos',
            font=dict(size=24, color='white', family='Inter'),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title='',
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=False,
            color='rgba(255, 255, 255, 0.7)'
        ),
        yaxis=dict(
            title='Monto del Dividendo (USD)',
            gridcolor='rgba(255, 255, 255, 0.05)',
            showgrid=True,
            zeroline=False,
            color='rgba(255, 255, 255, 0.7)',
            tickprefix='$'
        ),
        plot_bgcolor='rgba(10, 14, 39, 0.5)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=450,
        font=dict(family='Inter', color='white'),
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


def plot_dividend_growth(annual_div_df, ticker):
    """Crea análisis premium de crecimiento de dividendos"""
    if annual_div_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Tendencia de Dividendos Anuales', 'Tasa de Crecimiento Interanual'),
        vertical_spacing=0.12,
        row_heights=[0.55, 0.45]
    )
    
    # Dividendos anuales con gradiente
    fig.add_trace(
        go.Scatter(
            x=annual_div_df['year'],
            y=annual_div_df['annual_dividend'],
            mode='lines+markers',
            name='Dividendo Anual',
            line=dict(color='#00ff88', width=4),
            marker=dict(size=12, color='#00ff88', line=dict(color='white', width=2)),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)',
            hovertemplate='<b>%{x}</b><br>Dividendo: $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Tasa de crecimiento
    if len(annual_div_df) > 1:
        growth = annual_div_df['annual_dividend'].pct_change() * 100
        colors = ['#00ff88' if x >= 0 else '#ff6b6b' for x in growth]
        
        fig.add_trace(
            go.Bar(
                x=annual_div_df['year'],
                y=growth,
                name='Crecimiento %',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(255, 255, 255, 0.2)', width=1)
                ),
                hovertemplate='<b>%{x}</b><br>Crecimiento: %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.05)',
        showgrid=True,
        color='rgba(255, 255, 255, 0.7)',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Año",
        gridcolor='rgba(255, 255, 255, 0.05)',
        showgrid=False,
        color='rgba(255, 255, 255, 0.7)',
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Dividendo (USD)",
        gridcolor='rgba(255, 255, 255, 0.05)',
        showgrid=True,
        zeroline=False,
        color='rgba(255, 255, 255, 0.7)',
        tickprefix='$',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Tasa de Crecimiento (%)",
        gridcolor='rgba(255, 255, 255, 0.05)',
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(255, 255, 255, 0.2)',
        color='rgba(255, 255, 255, 0.7)',
        ticksuffix='%',
        row=2, col=1
    )
    
    fig.update_layout(
        height=650,
        plot_bgcolor='rgba(10, 14, 39, 0.5)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        font=dict(family='Inter', color='white', size=12),
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    return fig


def main():
    # Encabezado
    st.markdown("<h1>💎 Geraldine Weiss</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Plataforma Profesional de Valoración por Dividendos y Estrategia de Inversión</p>", unsafe_allow_html=True)
    
    # Barra lateral
    with st.sidebar:
        st.markdown("### ⚙️ Configuración")
        
        ticker = st.text_input("Ticker de la Acción", value="KO", help="Introduce el símbolo de una acción que pague dividendos")
        years = st.slider("Período de Análisis", 3, 10, 6, help="Años de datos históricos a analizar")
        
        st.markdown("---")
        
        analyze_button = st.button("🔍 Analizar Acción", type="primary", use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("""
        <div class='info-box'>
        <h4 style='margin-top: 0; color: #00d4ff;'>💡 Sobre Este Método</h4>
        <p style='font-size: 13px; line-height: 1.6; margin-bottom: 0;'>
        El enfoque de <strong>Geraldine Weiss</strong> identifica valor mediante análisis de rentabilidad por dividendo:
        </p>
        <ul style='font-size: 13px; line-height: 1.8; margin-top: 10px;'>
        <li><strong style='color: #00ff88;'>Alta rentabilidad</strong> = Infravalorada (Compra)</li>
        <li><strong style='color: #ff6b6b;'>Baja rentabilidad</strong> = Sobrevalorada (Venta)</li>
        <li><strong style='color: #ffd93d;'>Rango medio</strong> = Valor razonable (Mantener)</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style='font-size: 12px; color: rgba(255, 255, 255, 0.5); margin-top: 30px;'>
        <p><strong>Candidatos Ideales:</strong></p>
        <p>KO · JNJ · PG · MMM<br>CAT · XOM · CVX · T</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Contenido principal
    if analyze_button and ticker:
        with st.spinner('🔄 Analizando datos del mercado...'):
            analyzer = GeraldineWeissAnalyzer(ticker.upper(), years)
            
            price_data = analyzer.fetch_price_data()
            dividend_data = analyzer.fetch_dividend_data()
            
            if price_data is None or price_data.empty:
                st.error("❌ No se pudieron obtener datos de precio. Por favor verifica el símbolo del ticker.")
                return
            
            if dividend_data.empty:
                st.error("❌ No hay datos de dividendos disponibles. Esta estrategia requiere acciones que paguen dividendos.")
                return
            
            annual_dividends = analyzer.calculate_annual_dividends(dividend_data)
            analysis_df = analyzer.calculate_valuation_bands(price_data, annual_dividends)
            
            if analysis_df is None or analysis_df.empty:
                st.error("❌ Datos insuficientes para calcular las bandas de valoración.")
                return
            
            signal, description = analyzer.get_current_signal(analysis_df)
            
            # Mensaje de éxito
            st.markdown(f"""
            <div class='success-box'>
            <h4 style='margin: 0; color: #00ff88;'>✅ Análisis Completado</h4>
            <p style='margin: 8px 0 0 0; font-size: 14px;'>Se analizó exitosamente <strong>{ticker.upper()}</strong> con {len(dividend_data)} pagos de dividendos durante {years} años</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Banner de señal
            signal_colors = {
                "COMPRA FUERTE": "#00ff88",
                "COMPRA": "#51cf66",
                "MANTENER": "#ffd93d",
                "VENTA": "#ff8787",
                "VENTA FUERTE": "#ff6b6b"
            }
            
            st.markdown(f"""
            <div class='signal-banner' style='border-color: {signal_colors.get(signal, "#ffffff")};'>
                <h2 class='signal-title' style='color: {signal_colors.get(signal, "#ffffff")};'>
                    {signal}
                </h2>
                <p class='signal-description' style='color: rgba(255, 255, 255, 0.9);'>
                    {description}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Métricas clave
            st.markdown("### 📊 Métricas Clave")
            
            latest = analysis_df.iloc[-1]
            current_price = latest['Close']
            current_yield = latest['div_yield'] * 100
            undervalued_price = latest['undervalued']
            overvalued_price = latest['overvalued']
            upside_to_undervalued = ((undervalued_price/current_price - 1) * 100)
            upside_to_overvalued = ((overvalued_price/current_price - 1) * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Precio Actual", f"${current_price:.2f}")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Rentabilidad por Dividendo", f"{current_yield:.2f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col3:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Zona Infravalorada", f"${undervalued_price:.2f}", 
                         delta=f"{upside_to_undervalued:.1f}%",
                         delta_color="inverse")
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col4:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Zona Sobrevalorada", f"${overvalued_price:.2f}",
                         delta=f"{upside_to_overvalued:.1f}%",
                         delta_color="normal")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Pestañas
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Análisis de Valoración", "💰 Historial de Dividendos", 
                                               "📊 Métricas de Crecimiento", "📚 Guía de Estrategia"])
            
            with tab1:
                st.plotly_chart(plot_geraldine_weiss(analysis_df, ticker.upper()), 
                               use_container_width=True)
                
                st.markdown("### 🎯 Interpretación del Gráfico")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class='info-box' style='border-left-color: #00ff88;'>
                    <h4 style='color: #00ff88; margin-top: 0;'>🟢 Zona Infravalorada</h4>
                    <p style='font-size: 14px; line-height: 1.7;'>
                    Cuando el precio se aproxima a la <strong>línea verde</strong>, la acción ofrece alta rentabilidad por dividendo respecto a su promedio histórico.
                    </p>
                    <p style='font-size: 14px; line-height: 1.7; margin-bottom: 0;'>
                    <strong>Acción:</strong> Considerar acumular acciones<br>
                    <strong>Perfil de Riesgo:</strong> Menor relativo al rango histórico<br>
                    <strong>Retorno Esperado:</strong> Apreciación de capital + dividendos
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class='warning-box' style='border-left-color: #ff6b6b;'>
                    <h4 style='color: #ff6b6b; margin-top: 0;'>🔴 Zona Sobrevalorada</h4>
                    <p style='font-size: 14px; line-height: 1.7;'>
                    Cuando el precio se aproxima a la <strong>línea roja</strong>, la acción ofrece baja rentabilidad por dividendo respecto a su promedio histórico.
                    </p>
                    <p style='font-size: 14px; line-height: 1.7; margin-bottom: 0;'>
                    <strong>Acción:</strong> Considerar toma de beneficios<br>
                    <strong>Perfil de Riesgo:</strong> Mayor relativo al rango histórico<br>
                    <strong>Retorno Esperado:</strong> Potencial alcista limitado
                    </p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab2:
                fig_div = plot_dividend_history(dividend_data, ticker.upper())
                if fig_div:
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    st.markdown("### 📊 Estadísticas de Dividendos")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_divs = len(dividend_data)
                    avg_div = dividend_data['amount'].mean()
                    latest_div = dividend_data.iloc[0]['amount']
                    total_paid = dividend_data['amount'].sum()
                    
                    with col1:
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        st.metric("Pagos Totales", f"{total_divs:,}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with col2:
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        st.metric("Pago Promedio", f"${avg_div:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with col3:
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        st.metric("Último Pago", f"${latest_div:.3f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                    with col4:
                        st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                        st.metric("Total Acumulado", f"${total_paid:.2f}")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.markdown("### 📅 Pagos de Dividendos Recientes")
                    recent = dividend_data.head(12).copy()
                    recent['ex_dividend_date'] = recent['ex_dividend_date'].dt.strftime('%Y-%m-%d')
                    recent.columns = ['Fecha Ex-Dividendo', 'Monto']
                    recent['Monto'] = recent['Monto'].apply(lambda x: f"${x:.3f}")
                    st.dataframe(recent, use_container_width=True, hide_index=True)
            
            with tab3:
                fig_growth = plot_dividend_growth(annual_dividends, ticker.upper())
                if fig_growth:
                    st.plotly_chart(fig_growth, use_container_width=True)
                    
                    if len(annual_dividends) > 1:
                        st.markdown("### 📈 Análisis de Crecimiento")
                        
                        cagr = ((annual_dividends['annual_dividend'].iloc[-1] / 
                                annual_dividends['annual_dividend'].iloc[0]) ** 
                               (1 / (len(annual_dividends) - 1)) - 1) * 100
                        
                        avg_growth = annual_dividends['annual_dividend'].pct_change().mean() * 100
                        years_data = len(annual_dividends)
                        latest_annual = annual_dividends['annual_dividend'].iloc[-1]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.metric("CAGR de Dividendos", f"{cagr:.2f}%")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                        with col2:
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.metric("Crecimiento Anual Promedio", f"{avg_growth:.2f}%")
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                        with col3:
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.metric("Años Analizados", years_data)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                        with col4:
                            st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                            st.metric("Último Dividendo Anual", f"${latest_annual:.2f}")
                            st.markdown("</div>", unsafe_allow_html=True)
            
            with tab4:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("""
                    ### 🎓 Resumen de la Estrategia
                    
                    **El Método de Geraldine Weiss** es un enfoque probado de valoración por dividendos que identifica oportunidades de compra y venta basándose en patrones históricos de rentabilidad por dividendo.
                    
                    #### Filosofía Central
                    
                    Los precios de las acciones fluctúan, pero las empresas de calidad mantienen pagos de dividendos estables. Esto crea patrones predecibles de rentabilidad que señalan valor.
                    
                    #### Principios Clave
                    
                    **Alta Rentabilidad por Dividendo** → Acción infravalorada  
                    **Baja Rentabilidad por Dividendo** → Acción sobrevalorada  
                    **Reversión a la Media** → Las rentabilidades vuelven al promedio histórico
                    
                    #### Implementación
                    
                    1. **Zona de Compra**: Entrar cuando el precio cruza al territorio infravalorado
                    2. **Mantener**: Conservar la posición mientras se cobran dividendos
                    3. **Zona de Venta**: Salir cuando el precio alcanza niveles sobrevalorados
                    4. **Repetir**: Reinvertir las ganancias en nuevas oportunidades infravaloradas
                    
                    """)
                
                with col2:
                    st.markdown("""
                    ### ✅ Candidatos Ideales
                    
                    <div class='success-box'>
                    <p style='margin: 0; font-size: 14px; line-height: 1.8;'>
                    <strong>Aristócratas de Dividendos</strong><br>
                    25+ años consecutivos de aumentos de dividendos
                    </p>
                    </div>
                    
                    <div class='success-box'>
                    <p style='margin: 0; font-size: 14px; line-height: 1.8;'>
                    <strong>Empresas Blue-Chip</strong><br>
                    Líderes del mercado establecidos con flujos de caja estables
                    </p>
                    </div>
                    
                    <div class='success-box'>
                    <p style='margin: 0; font-size: 14px; line-height: 1.8;'>
                    <strong>Pagadores Consistentes</strong><br>
                    Dividendos trimestrales regulares sin recortes
                    </p>
                    </div>
                    
                    ### ⚠️ Consideraciones de Riesgo
                    
                    <div class='warning-box'>
                    <ul style='margin: 0; font-size: 14px; line-height: 1.8; padding-left: 20px;'>
                    <li>Verificar sostenibilidad del dividendo (payout ratio < 60%)</li>
                    <li>Monitorear continuamente los fundamentos de la empresa</li>
                    <li>Diversificar entre múltiples sectores</li>
                    <li>Evitar industrias cíclicas o volátiles</li>
                    <li>Nunca depender de una única métrica de valoración</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                st.markdown("""
                ### 📊 Rendimiento Histórico
                
                La estrategia de Geraldine Weiss ha entregado históricamente:
                
                - **15-20% de descuento** de entrada al valor razonable
                - **15-20% de prima** de salida al valor razonable  
                - **Ingresos por dividendos consistentes** durante todo el período de tenencia
                - **Apreciación de capital a largo plazo** de empresas de calidad
                - **Menor volatilidad** comparada con estrategias enfocadas en crecimiento
                
                ---
                
                <div style='background: rgba(255, 107, 107, 0.1); border: 1px solid rgba(255, 107, 107, 0.3); border-radius: 12px; padding: 20px; margin-top: 30px;'>
                <p style='margin: 0; font-size: 13px; color: rgba(255, 255, 255, 0.8); line-height: 1.6;'>
                <strong style='color: #ff6b6b;'>⚠️ Aviso Legal:</strong> Esta herramienta es solo para fines educativos e informativos. 
                No constituye asesoramiento financiero, recomendaciones de inversión ni una oferta de compra o venta de valores. 
                Siempre realiza una investigación exhaustiva y consulta con un asesor financiero cualificado antes de tomar decisiones de inversión. 
                El rendimiento pasado no garantiza resultados futuros.
                </p>
                </div>
                """, unsafe_allow_html=True)
    
    else:
        # Pantalla de bienvenida
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            <div class='welcome-container'>
                <h2 style='margin-top: 0; font-size: 36px;'>👋 Bienvenido</h2>
                <p style='font-size: 18px; color: rgba(255, 255, 255, 0.8); line-height: 1.8; margin: 20px 0;'>
                Una plataforma profesional que implementa la legendaria metodología de valoración por dividendos de <strong>Geraldine Weiss</strong>.
                </p>
                
                <div style='background: rgba(0, 212, 255, 0.1); border-radius: 12px; padding: 30px; margin: 30px 0;'>
                    <h3 style='margin-top: 0; color: #00d4ff;'>🎯 Lo Que Ofrece Esta Herramienta</h3>
                    <ul style='text-align: left; font-size: 15px; line-height: 2; color: rgba(255, 255, 255, 0.9);'>
                        <li>Análisis histórico de rentabilidad por dividendo</li>
                        <li>Bandas dinámicas de valoración (sobrevalorada/infravalorada)</li>
                        <li>Señales claras de compra/venta/mantener</li>
                        <li>Seguimiento completo de pagos de dividendos</li>
                        <li>Cálculos de tasa de crecimiento y CAGR</li>
                        <li>Visualizaciones interactivas</li>
                    </ul>
                </div>
                
                <p style='font-size: 16px; margin: 30px 0;'>
                Introduce un símbolo ticker en la barra lateral para comenzar tu análisis
                </p>
                
                <div style='background: rgba(0, 255, 136, 0.05); border-radius: 8px; padding: 16px; margin-top: 20px;'>
                    <p style='margin: 0; font-size: 14px; color: rgba(255, 255, 255, 0.7);'>
                    <strong>Tickers Populares:</strong> KO · JNJ · PG · MMM · CAT · XOM · CVX
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Créditos del autor (siempre visible)
    st.markdown("""
    <div class='author-credit'>
        Desarrollado por <a href='https://bquantfinance.com' target='_blank'>@Gsnchez | bquantfinance.com</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
