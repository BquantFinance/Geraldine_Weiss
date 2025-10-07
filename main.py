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
    
    .comparison-card {
        background-color: #1a1f2e;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin: 10px 0;
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
        
    def is_european_ticker(self, ticker: str) -> bool:
        """Detecta si un ticker es europeo basándose en sufijos comunes"""
        european_suffixes = [
            '.MC', '.PA', '.DE', '.L', '.AS', '.MI', '.BR', '.HE', 
            '.OL', '.ST', '.CO', '.LS', '.AT', '.PR', '.VI', '.IR', '.WA',
        ]
        return any(ticker.upper().endswith(suffix) for suffix in european_suffixes)
        
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
        """Obtiene datos de dividendos con lógica inteligente de fuentes"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        is_european = self.is_european_ticker(self.ticker)
        
        if not is_european:
            df = self.dividend_fetcher.fetch_dividends(
                self.ticker, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if not df.empty:
                return df
        
        try:
            ticker_obj = yf.Ticker(self.ticker)
            divs = ticker_obj.dividends
            
            if not divs.empty:
                df = pd.DataFrame({
                    'ex_dividend_date': divs.index,
                    'amount': divs.values
                })
                df = df[df['ex_dividend_date'] >= start_date]
                
                if not df.empty:
                    return df
        except:
            pass
        
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
        
        # Calcular score numérico (-100 a +100)
        # -100 = muy sobrevalorada, +100 = muy infravalorada
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
    """Análisis rápido de un ticker para comparación"""
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
        
        # Calcular CAGR si hay suficientes datos
        cagr = 0
        if len(annual_dividends) > 1:
            cagr = ((annual_dividends['annual_dividend'].iloc[-1] / 
                    annual_dividends['annual_dividend'].iloc[0]) ** 
                   (1 / (len(annual_dividends) - 1)) - 1) * 100
        
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
    except Exception as e:
        # En caso de error, retornar None
        return None


def plot_comparison_chart(results):
    """Gráfico comparativo de múltiples tickers"""
    fig = go.Figure()
    
    tickers = [r['ticker'] for r in results]
    prices = [r['price'] for r in results]
    undervalued = [r['undervalued'] for r in results]
    overvalued = [r['overvalued'] for r in results]
    
    x = list(range(len(tickers)))
    
    # Bandas
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
    
    # Precio actual
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
        text=[f"${p:.2f}" for p in prices],
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
            title='Precio (USD)',
            tickprefix='$'
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
    
    labels = [f"{row['ticker']}<br>{row['weight']}%" for _, row in portfolio_data.iterrows()]
    values = portfolio_data['weight'].tolist()
    
    colors = ['#00ff88', '#00d4ff', '#7b2ff7', '#ffd93d', '#ff6b6b', 
              '#51cf66', '#ff8787', '#00b4d8', '#90e0ef', '#ff006e']
    
    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors[:len(labels)], line=dict(color='#1a1f2e', width=2)),
        textinfo='label+percent',
        textfont=dict(size=14),
        hovertemplate='<b>%{label}</b><br>Peso: %{value}%<extra></extra>'
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
    
    # Tabs principales
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
                value=6,
                help="Años de datos históricos a analizar"
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
                
                ℹ️ **Fuentes de datos:**
                - Acciones USA: dividendhistory.org
                - Acciones Europa: yfinance
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
            st.caption("**USA:** KO · JNJ · PG · MMM · CAT · XOM · CVX")
            st.caption("**Europa:** IBE.MC · SAN.MC · TEF.MC · REP.MC")
        
        with col_main:
            if analyze_button and ticker:
                with st.spinner('🔄 Analizando datos del mercado...'):
                    result = analyze_ticker_quick(ticker.upper(), years)
                    
                    if result is None:
                        st.error(f"""
                        ❌ **No se pudieron obtener datos suficientes para {ticker.upper()}**
                        
                        Posibles causas:
                        - El ticker no existe o está mal escrito
                        - La acción no paga dividendos
                        - No hay suficiente historial de datos ({years} años)
                        - Problemas de conexión con las fuentes de datos
                        
                        💡 **Sugerencias:**
                        - Verifica que el ticker sea correcto (ej: KO, JNJ, PG)
                        - Para acciones europeas usa el sufijo: IBE.MC, SAN.MC
                        - Reduce el período de análisis a 3 años
                        - Prueba con otra acción que pague dividendos regularmente
                        """)
                    else:
                        # Mensaje de éxito
                        st.success(f"✅ Análisis completado para **{ticker.upper()}**")
                        
                        # Señal principal
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
                            <div style='font-size: 18px; margin-top: 10px;'>Precio: ${result['price']:.2f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        # Métricas clave
                        st.subheader("📊 Métricas Clave")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        upside_to_undervalued = ((result['undervalued']/result['price'] - 1) * 100)
                        upside_to_overvalued = ((result['overvalued']/result['price'] - 1) * 100)
                        
                        col1.metric("💵 Precio Actual", f"${result['price']:.2f}")
                        col2.metric("📊 Rentabilidad", f"{result['yield']:.2f}%")
                        col3.metric("🟢 Zona Infravalorada", f"${result['undervalued']:.2f}", 
                                   delta=f"{upside_to_undervalued:.1f}%", delta_color="inverse")
                        col4.metric("🔴 Zona Sobrevalorada", f"${result['overvalued']:.2f}",
                                   delta=f"{upside_to_overvalued:.1f}%")
                        
                        st.divider()
                        
                        # Mostrar análisis detallado (código existente)
                        st.info("💡 Para análisis detallado completo, consulta la versión anterior del código")
            else:
                st.info("""
                ### 👋 Bienvenido al Analizador Geraldine Weiss
                
                Introduce un ticker en la barra lateral para comenzar el análisis individual.
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
                        st.error("""
                        ❌ **No se pudieron obtener datos para ningún ticker**
                        
                        Verifica que:
                        - Los tickers sean correctos
                        - Las acciones paguen dividendos
                        - Haya suficiente historial
                        """)
                    else:
                    st.success(f"✅ Análisis completado para {len(results)} acciones")
                    
                    # Gráfico comparativo
                    st.plotly_chart(plot_comparison_chart(results), use_container_width=True)
                    
                    st.divider()
                    
                    # Tabla comparativa
                    st.subheader("📋 Tabla Comparativa")
                    
                    comparison_df = pd.DataFrame([{
                        'Ticker': r['ticker'],
                        'Precio': f"${r['price']:.2f}",
                        'Yield': f"{r['yield']:.2f}%",
                        'Infravalorada': f"${r['undervalued']:.2f}",
                        'Sobrevalorada': f"${r['overvalued']:.2f}",
                        'Señal': r['signal'],
                        'Score': f"{r['score']:.1f}",
                        'CAGR Div.': f"{r['cagr']:.1f}%"
                    } for r in results])
                    
                    # Ordenar por score (mejores oportunidades primero)
                    comparison_df = comparison_df.sort_values('Score', ascending=False)
                    
                    st.dataframe(
                        comparison_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                            "Señal": st.column_config.TextColumn("Señal", width="medium"),
                            "Score": st.column_config.TextColumn("Score", width="small", help="Score: +100=muy infravalorada, -100=muy sobrevalorada")
                        }
                    )
                    
                    # Ranking de oportunidades
                    st.divider()
                    st.subheader("🏆 Ranking de Oportunidades")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    # Top compra
                    buy_opportunities = [r for r in results if 'COMPRA' in r['signal']]
                    buy_opportunities.sort(key=lambda x: x['score'], reverse=True)
                    
                    with col1:
                        st.success("**🟢 Mejores Oportunidades de Compra**")
                        if buy_opportunities:
                            for r in buy_opportunities[:3]:
                                st.markdown(f"**{r['ticker']}** - {r['signal']}")
                                st.caption(f"Yield: {r['yield']:.2f}% | Score: {r['score']:.1f}")
                        else:
                            st.caption("No hay señales de compra")
                    
                    # Mantener
                    hold_opportunities = [r for r in results if r['signal'] == 'MANTENER']
                    
                    with col2:
                        st.info("**🟡 Mantener Posición**")
                        if hold_opportunities:
                            for r in hold_opportunities[:3]:
                                st.markdown(f"**{r['ticker']}** - {r['signal']}")
                                st.caption(f"Yield: {r['yield']:.2f}%")
                        else:
                            st.caption("No hay señales de mantener")
                    
                    # Top venta
                    sell_opportunities = [r for r in results if 'VENTA' in r['signal']]
                    sell_opportunities.sort(key=lambda x: x['score'])
                    
                    with col3:
                        st.warning("**🔴 Considerar Venta**")
                        if sell_opportunities:
                            for r in sell_opportunities[:3]:
                                st.markdown(f"**{r['ticker']}** - {r['signal']}")
                                st.caption(f"Yield: {r['yield']:.2f}% | Score: {r['score']:.1f}")
                        else:
                            st.caption("No hay señales de venta")
    
    # ==================== TAB 3: CARTERA PONDERADA ====================
    with main_tab3:
        st.header("💼 Análisis de Cartera Ponderada")
        st.caption("Aplica el método Geraldine Weiss a tu cartera completa")
        
        # Inicializar session state para la cartera
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = pd.DataFrame(columns=['ticker', 'weight'])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("➕ Añadir Posiciones")
            
            col_a, col_b, col_c = st.columns([2, 1, 1])
            
            with col_a:
                new_ticker = st.text_input("Ticker", key="portfolio_ticker")
            with col_b:
                new_weight = st.number_input("Peso (%)", min_value=0.0, max_value=100.0, value=10.0, step=5.0, key="portfolio_weight")
            with col_c:
                st.write("")
                st.write("")
                if st.button("➕ Añadir", type="secondary", use_container_width=True):
                    if new_ticker:
                        new_row = pd.DataFrame([{'ticker': new_ticker.upper(), 'weight': new_weight}])
                        st.session_state.portfolio = pd.concat([st.session_state.portfolio, new_row], ignore_index=True)
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
        
        # Mostrar cartera actual
        if not st.session_state.portfolio.empty:
            st.subheader("📊 Cartera Actual")
            
            # Normalizar pesos
            total_weight = st.session_state.portfolio['weight'].sum()
            if total_weight != 100:
                st.warning(f"⚠️ Los pesos suman {total_weight:.1f}%. Se normalizarán a 100%.")
            
            # Mostrar tabla editable
            edited_df = st.data_editor(
                st.session_state.portfolio,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                column_config={
                    "ticker": st.column_config.TextColumn("Ticker", width="medium"),
                    "weight": st.column_config.NumberColumn("Peso (%)", width="small", min_value=0, max_value=100)
                }
            )
            
            st.session_state.portfolio = edited_df
            
            st.divider()
            
            # Botón de análisis
            if st.button("🔍 Analizar Cartera Completa", type="primary", use_container_width=True):
                with st.spinner('🔄 Analizando cartera...'):
                    # Normalizar pesos
                    portfolio_data = st.session_state.portfolio.copy()
                    portfolio_data['weight'] = (portfolio_data['weight'] / portfolio_data['weight'].sum()) * 100
                    
                    # Analizar cada ticker
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
                        st.error("""
                        ❌ **No se pudieron obtener datos para ningún ticker de la cartera**
                        
                        Verifica que:
                        - Los tickers sean correctos
                        - Las acciones paguen dividendos
                        - Haya conexión a internet
                        """)
                    else:
                        st.success(f"✅ Cartera analizada: {len(portfolio_results)} posiciones")
                        
                        # Calcular métricas de cartera ponderada
                        total_yield = sum(r['yield'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        total_cagr = sum(r['cagr'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        avg_score = sum(r['score'] * r['portfolio_weight'] / 100 for r in portfolio_results)
                        
                        # Determinar señal de cartera
                        if avg_score > 30:
                            portfolio_signal = "COMPRA"
                            signal_color = "#00ff88"
                        elif avg_score < -30:
                            portfolio_signal = "VENTA"
                            signal_color = "#ff6b6b"
                        else:
                            portfolio_signal = "MANTENER"
                            signal_color = "#ffd93d"
                        
                        # Banner de señal de cartera
                        st.markdown(
                            f"""<div class='big-signal' style='border-color: {signal_color}; color: {signal_color}; font-size: 36px;'>
                            Señal de Cartera: {portfolio_signal}<br>
                            <div style='font-size: 16px; margin-top: 10px;'>Score Ponderado: {avg_score:.1f}</div>
                            </div>""",
                            unsafe_allow_html=True
                        )
                        
                        # Métricas de cartera
                        st.subheader("📊 Métricas de Cartera Ponderada")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric("💰 Yield Promedio", f"{total_yield:.2f}%")
                        col2.metric("📈 CAGR Promedio", f"{total_cagr:.2f}%")
                        col3.metric("🎯 Score Cartera", f"{avg_score:.1f}")
                        col4.metric("📋 Posiciones", len(portfolio_results))
                        
                        st.divider()
                        
                        # Composición
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.plotly_chart(
                                plot_portfolio_composition(portfolio_data),
                                use_container_width=True
                            )
                        
                        with col2:
                            st.subheader("📋 Detalle de Posiciones")
                            
                            portfolio_detail = pd.DataFrame([{
                                'Ticker': r['ticker'],
                                'Peso': f"{r['portfolio_weight']:.1f}%",
                                'Yield': f"{r['yield']:.2f}%",
                                'Señal': r['signal'],
                                'Score': f"{r['score']:.1f}"
                            } for r in portfolio_results])
                            
                            st.dataframe(
                                portfolio_detail,
                                use_container_width=True,
                                hide_index=True
                            )
                        
                        st.divider()
                        
                        # Recomendaciones
                        st.subheader("💡 Recomendaciones de Rebalanceo")
                        
                        buy_positions = [r for r in portfolio_results if 'COMPRA' in r['signal']]
                        sell_positions = [r for r in portfolio_results if 'VENTA' in r['signal']]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success("**🟢 Considerar Aumentar Posición**")
                            if buy_positions:
                                for r in buy_positions:
                                    st.markdown(f"**{r['ticker']}** ({r['portfolio_weight']:.1f}%) - {r['signal']}")
                                    st.caption(f"Yield: {r['yield']:.2f}% | Score: {r['score']:.1f}")
                            else:
                                st.caption("No hay posiciones en zona de compra")
                        
                        with col2:
                            st.warning("**🔴 Considerar Reducir Posición**")
                            if sell_positions:
                                for r in sell_positions:
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
    
    # Créditos
    st.markdown("""
    <div class='footer-credit'>
        Desarrollado por <a href='https://bquantfinance.com' target='_blank'>@Gsnchez | bquantfinance.com</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
