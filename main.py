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
import warnings
import pytz

# Suprimir warnings de yfinance
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
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }
            
            response = self.session.get(url, headers=headers, timeout=20)
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
                df['amount'] = df['amount'].astype(str).str.replace('$', '', regex=False)
                df['amount'] = df['amount'].str.replace(',', '', regex=False).str.strip()
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
        
    def fetch_price_data(self):
        """Obtiene datos históricos de precios"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        try:
            data = yf.download(
                self.ticker, 
                start=start_date, 
                end=end_date, 
                progress=False,
                auto_adjust=True
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
        """Obtiene datos de dividendos con lógica clara de fuentes"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        has_suffix = '.' in self.ticker
        
        if not has_suffix:
            st.info(f"🇺🇸 Ticker USA detectado ({self.ticker}). Usando dividendhistory.org...")
            try:
                df = self.dividend_fetcher.fetch_dividends(
                    self.ticker, 
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if not df.empty:
                    st.success(f"✅ Encontrados {len(df)} pagos desde dividendhistory.org")
                    return df
                else:
                    st.warning(f"⚠️ dividendhistory.org no retornó datos. Fallback a yfinance...")
                    return self._fetch_from_yfinance(start_date)
            except Exception as e:
                st.warning(f"⚠️ Error con dividendhistory.org. Fallback a yfinance...")
                return self._fetch_from_yfinance(start_date)
        else:
            st.info(f"🌍 Ticker internacional ({self.ticker}). Usando yfinance...")
            return self._fetch_from_yfinance(start_date)
    
    def _fetch_from_yfinance(self, start_date):
        """Helper para obtener dividendos desde yfinance"""
        try:
            ticker_obj = yf.Ticker(self.ticker)
            divs = ticker_obj.dividends
            
            if not divs.empty:
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
                    df = df.sort_values('ex_dividend_date', ascending=False).reset_index(drop=True)
                    st.success(f"✅ Encontrados {len(df)} pagos de dividendos desde yfinance")
                    return df
                else:
                    st.error(f"❌ No hay dividendos en el período de {self.years} años")
                    return pd.DataFrame()
            else:
                st.error(f"❌ yfinance no retornó dividendos para {self.ticker}")
                return pd.DataFrame()
        except Exception as e:
            st.error(f"❌ Error con yfinance: {str(e)}")
            return pd.DataFrame()
    
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
        
        if 'Close' not in prices.columns:
            if 'close' in prices.columns:
                prices = prices.rename(columns={'close': 'Close'})
            else:
                return None
        
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
            return "COMPRA FUERTE", f"Precio ${price:.2f} está en zona infravalorada", score
        elif price <= undervalued:
            return "COMPRA", f"Precio ${price:.2f} se aproxima al nivel infravalorado", score
        elif price >= upper_sell_zone:
            return "VENTA FUERTE", f"Precio ${price:.2f} está en zona sobrevalorada", score
        elif price >= overvalued:
            return "VENTA", f"Precio ${price:.2f} se aproxima al nivel sobrevalorado", score
        else:
            return "MANTENER", f"Precio ${price:.2f} está en rango de valor razonable", score


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
        if annual_dividends.empty:
            return None
        
        analysis_df = analyzer.calculate_valuation_bands(price_data, annual_dividends)
        
        if analysis_df is None or analysis_df.empty:
            return None
        
        signal, description, score = analyzer.get_current_signal(analysis_df)
        latest = analysis_df.iloc[-1]
        
        cagr = 0
        if len(annual_dividends) > 1:
            try:
                cagr = ((annual_dividends['annual_dividend'].iloc[-1] / 
                        annual_dividends['annual_dividend'].iloc[0]) ** 
                       (1 / (len(annual_dividends) - 1)) - 1) * 100
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
            'dividend_data': dividend_data
        }
    except Exception:
        return None


def plot_geraldine_weiss_individual(analysis_df, ticker):
    """Gráfico de Geraldine Weiss para análisis individual"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index, y=analysis_df['overvalued'],
        name='Sobrevalorada', fill='tonexty', fillcolor='rgba(255, 107, 107, 0.1)',
        line=dict(color='#ff6b6b', width=3), mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index, y=analysis_df['undervalued'],
        name='Infravalorada', line=dict(color='#00ff88', width=3), mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=analysis_df.index, y=analysis_df['Close'],
        name='Precio Actual', line=dict(color='#00d4ff', width=4), mode='lines'
    ))
    
    latest = analysis_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[analysis_df.index[-1]], y=[latest['Close']], mode='markers',
        marker=dict(size=16, color='#00d4ff', line=dict(color='white', width=3)),
        showlegend=False
    ))
    
    fig.update_layout(
        title=f'{ticker} - Modelo de Valoración Geraldine Weiss',
        xaxis_title='Fecha', yaxis_title='Precio (USD)',
        template='plotly_dark', height=550,
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        hovermode='x unified'
    )
    
    return fig


def plot_comparison_chart(results):
    """Gráfico comparativo"""
    fig = go.Figure()
    
    tickers = [r['ticker'] for r in results]
    x = list(range(len(tickers)))
    
    fig.add_trace(go.Scatter(
        x=x, y=[r['overvalued'] for r in results],
        name='Sobrevalorada', line=dict(color='#ff6b6b', width=2, dash='dash'),
        mode='lines+markers'
    ))
    
    fig.add_trace(go.Scatter(
        x=x, y=[r['undervalued'] for r in results],
        name='Infravalorada', line=dict(color='#00ff88', width=2, dash='dash'),
        mode='lines+markers'
    ))
    
    colors = ['#00ff88' if 'COMPRA' in r['signal'] else '#ff6b6b' if 'VENTA' in r['signal'] else '#ffd93d' for r in results]
    
    fig.add_trace(go.Scatter(
        x=x, y=[r['price'] for r in results],
        name='Precio Actual', mode='lines+markers',
        line=dict(color='#00d4ff', width=4),
        marker=dict(size=15, color=colors, line=dict(color='white', width=2))
    ))
    
    fig.update_layout(
        title='Comparación - Método Geraldine Weiss',
        xaxis=dict(tickmode='array', tickvals=x, ticktext=tickers),
        yaxis=dict(title='Precio (USD)', tickprefix='$'),
        template='plotly_dark', height=500,
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117'
    )
    
    return fig


def plot_portfolio_composition(portfolio_data):
    """Gráfico de composición de cartera"""
    fig = go.Figure()
    
    labels = [f"{row['ticker']}<br>{row['weight']:.1f}%" for _, row in portfolio_data.iterrows()]
    colors = ['#00ff88', '#00d4ff', '#7b2ff7', '#ffd93d', '#ff6b6b', '#51cf66']
    
    fig.add_trace(go.Pie(
        labels=labels, values=portfolio_data['weight'].tolist(),
        marker=dict(colors=colors[:len(labels)])
    ))
    
    fig.update_layout(
        title='Composición de la Cartera',
        template='plotly_dark', height=450,
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117'
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
    
    # TAB 1: ANÁLISIS INDIVIDUAL
    with main_tab1:
        col_sidebar, col_main = st.columns([1, 3])
        
        with col_sidebar:
            st.header("⚙️ Configuración")
            ticker = st.text_input("Stock Ticker", value="KO")
            years = st.slider("Período (años)", 3, 10, 6)
            st.divider()
            analyze_button = st.button("🔍 Analizar Acción", type="primary", use_container_width=True)
            st.divider()
            
            with st.expander("💡 Sobre Este Método"):
                st.markdown("""
                **Geraldine Weiss** identifica valor mediante rentabilidad por dividendo:
                - Alta rentabilidad = Infravalorada (Compra)
                - Baja rentabilidad = Sobrevalorada (Venta)
                """)
            
            st.markdown("**🎯 Tickers Sugeridos**")
            st.caption("USA: KO · JNJ · PG · PEP")
            st.caption("Europa: IBE.MC · SAN.MC")
        
        with col_main:
            if analyze_button and ticker:
                with st.spinner('🔄 Analizando...'):
                    result = analyze_ticker_quick(ticker.upper(), years)
                    
                    if result is None:
                        st.error(f"❌ No se pudieron obtener datos para {ticker.upper()}")
                    else:
                        st.success(f"✅ Análisis completado para {ticker.upper()}")
                        
                        signal_colors = {
                            "COMPRA FUERTE": "#00ff88", "COMPRA": "#51cf66",
                            "MANTENER": "#ffd93d", "VENTA": "#ff8787",
                            "VENTA FUERTE": "#ff6b6b"
                        }
                        
                        st.markdown(
                            f"""<div class='big-signal' style='border-color: {signal_colors.get(result['signal'], "#fff")}; color: {signal_colors.get(result['signal'], "#fff")}'>
                            {result['signal']}<br>
                            <div style='font-size: 18px; margin-top: 10px;'>Precio: ${result['price']:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        st.subheader("📊 Métricas Clave")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        upside_u = ((result['undervalued']/result['price'] - 1) * 100)
                        upside_o = ((result['overvalued']/result['price'] - 1) * 100)
                        
                        col1.metric("💵 Precio", f"${result['price']:.2f}")
                        col2.metric("📊 Yield", f"{result['yield']:.2f}%")
                        col3.metric("🟢 Infravalorada", f"${result['undervalued']:.2f}", delta=f"{upside_u:.1f}%", delta_color="inverse")
                        col4.metric("🔴 Sobrevalorada", f"${result['overvalued']:.2f}", delta=f"{upside_o:.1f}%")
                        
                        st.divider()
                        st.subheader("📈 Análisis de Valoración")
                        st.plotly_chart(plot_geraldine_weiss_individual(result['analysis_df'], ticker.upper()), use_container_width=True)
            else:
                st.info("👋 Introduce un ticker y haz clic en **Analizar Acción**")
    
    # TAB 2: COMPARACIÓN
    with main_tab2:
        st.header("📊 Comparación Multi-Ticker")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            tickers_input = st.text_input("Tickers (separados por comas)", value="KO, PG, JNJ, PEP")
        with col2:
            years_comp = st.selectbox("Período", [3, 5, 6, 10], index=2)
            compare_button = st.button("🔍 Comparar", type="primary", use_container_width=True)
        
        if compare_button:
            tickers_list = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]
            
            if len(tickers_list) < 2:
                st.error("❌ Mínimo 2 tickers")
            else:
                with st.spinner(f'🔄 Analizando {len(tickers_list)} acciones...'):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, ticker in enumerate(tickers_list[:6]):
                        result = analyze_ticker_quick(ticker, years_comp)
                        if result:
                            results.append(result)
                        progress_bar.progress((i + 1) / min(len(tickers_list), 6))
                    
                    progress_bar.empty()
                    
                    if results:
                        st.success(f"✅ {len(results)} acciones analizadas")
                        st.plotly_chart(plot_comparison_chart(results), use_container_width=True)
                        
                        st.subheader("📋 Tabla Comparativa")
                        comparison_df = pd.DataFrame([{
                            'Ticker': r['ticker'], 'Precio': f"${r['price']:.2f}",
                            'Yield': f"{r['yield']:.2f}%", 'Señal': r['signal'],
                            'Score': f"{r['score']:.1f}"
                        } for r in results])
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # TAB 3: CARTERA
    with main_tab3:
        st.header("💼 Cartera Ponderada")
        
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'weight'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("➕ Añadir Posiciones")
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                new_ticker = st.text_input("Ticker", key="portfolio_ticker")
            with col_b:
                new_weight = st.number_input("Peso (%)", 0.0, 100.0, 10.0, 5.0, key="portfolio_weight")
            with col_c:
                st.write(""); st.write("")
                if st.button("➕ Añadir", type="secondary", use_container_width=True):
                    if new_ticker:
                        new_row = pd.DataFrame([{'ticker': new_ticker.upper(), 'weight': new_weight}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
                        st.rerun()
        
        with col2:
            st.subheader("🎯 Acciones Rápidas")
            if st.button("🗑️ Limpiar", use_container_width=True):
                st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'weight'])
                st.rerun()
            
            if st.button("📋 Ejemplo", use_container_width=True):
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
            edited_df = st.data_editor(st.session_state.portfolio, use_container_width=True, hide_index=True, num_rows="dynamic")
            st.session_state.portfolio = edited_df
            
            st.divider()
            
            if st.button("🔍 Analizar Cartera", type="primary", use_container_width=True):
                with st.spinner('🔄 Analizando cartera...'):
                    portfolio_data = st.session_state.portfolio.copy()
                    portfolio_data['weight'] = (portfolio_data['weight'] / portfolio_data['weight'].sum()) * 100
                    
                    portfolio_results = []
                    for _, row in portfolio_data.iterrows():
                        result = analyze_ticker_quick(row['ticker'], 6)
                        if result:
                            result['portfolio_weight'] = row['weight']
                            portfolio_results.append(result)
                    
                    if portfolio_results:
                        total_yield = sum(r['yield'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        avg_score = sum(r['score'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        
                        signal_color = "#00ff88" if avg_score > 30 else "#ff6b6b" if avg_score < -30 else "#ffd93d"
                        portfolio_signal = "COMPRA" if avg_score > 30 else "VENTA" if avg_score < -30 else "MANTENER"
                        
                        st.markdown(
                            f"""<div class='big-signal' style='border-color: {signal_color}; color: {signal_color}; font-size: 36px;'>
                            Señal: {portfolio_signal}<br>
                            <div style='font-size: 16px; margin-top: 10px;'>Score: {avg_score:.1f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        col1, col2 = st.columns(2)
                        col1.metric("💰 Yield Promedio", f"{total_yield:.2f}%")
                        col2.metric("📋 Posiciones", len(portfolio_results))
                        
                        st.plotly_chart(plot_portfolio_composition(portfolio_data), use_container_width=True)
        else:
            st.info("➕ Añade posiciones para comenzar")
    
    st.markdown("""
    <div class='footer-credit'>
        Desarrollado por <a href='https://bquantfinance.com' target='_blank'>@Gsnchez | bquantfinance.com</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
