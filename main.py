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
            
            # Aplanar MultiIndex si existe
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
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
        
        # Asegurarse de que tenemos la columna 'Close'
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
    """Crea gráfico de Geraldine Weiss"""
    fig = go.Figure()
    
    # Área de relleno entre bandas
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
    
    # Línea de sobrevaloración
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['overvalued'],
        name='Sobrevalorada',
        line=dict(color='#ff6b6b', width=3, dash='solid'),
        mode='lines',
        hovertemplate='<b>Sobrevalorada:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Línea de infravaloración
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['undervalued'],
        name='Infravalorada',
        line=dict(color='#00ff88', width=3, dash='solid'),
        mode='lines',
        hovertemplate='<b>Infravalorada:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Precio actual con gradiente
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['Close'],
        name='Precio Actual',
        line=dict(color='#00d4ff', width=4),
        mode='lines',
        hovertemplate='<b>Precio:</b> $%{y:.2f}<extra></extra>'
    ))
    
    # Marcador para el precio actual
    latest = analysis_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[analysis_df.index[-1]],
        y=[latest['Close']],
        mode='markers',
        marker=dict(
            size=16,
            color='#00d4ff',
            line=dict(color='white', width=3),
            symbol='circle'
        ),
        showlegend=False,
        hovertemplate=f'<b>Precio Actual:</b> ${latest["Close"]:.2f}<extra></extra>'
    ))
    
    # Añadir anotación para el precio actual
    fig.add_annotation(
        x=analysis_df.index[-1],
        y=latest['Close'],
        text=f"${latest['Close']:.2f}",
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
            zeroline=False
        ),
        yaxis=dict(
            title='Precio (USD)',
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False,
            tickprefix='$'
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


def plot_dividend_history(dividend_df, ticker):
    """Crea gráfico del historial de dividendos"""
    if dividend_df.empty:
        return None
    
    fig = go.Figure()
    
    # Crear gradiente de colores más sofisticado
    n = len(dividend_df)
    colors = []
    for i in range(n):
        # Gradiente de verde a cyan
        ratio = i / max(n - 1, 1)
        colors.append(f'rgba({int(0 + 0 * ratio)}, {int(255 - 43 * ratio)}, {int(136 + 119 * ratio)}, 0.8)')
    
    fig.add_trace(go.Bar(
        x=dividend_df['ex_dividend_date'],
        y=dividend_df['amount'],
        name='Dividendo',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1)
        ),
        hovertemplate='<b>Fecha:</b> %{x|%Y-%m-%d}<br><b>Monto:</b> $%{y:.3f}<extra></extra>'
    ))
    
    # Añadir línea de tendencia
    if len(dividend_df) > 1:
        fig.add_trace(go.Scatter(
            x=dividend_df['ex_dividend_date'],
            y=dividend_df['amount'].rolling(window=4, min_periods=1).mean(),
            name='Tendencia',
            line=dict(color='#ffd93d', width=2, dash='dash'),
            mode='lines',
            hovertemplate='<b>Media móvil:</b> $%{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'{ticker} - Historial de Pagos de Dividendos',
            font=dict(size=20, color='white')
        ),
        xaxis=dict(
            title='Fecha',
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Monto del Dividendo (USD)',
            gridcolor='rgba(255, 255, 255, 0.1)',
            showgrid=True,
            zeroline=False,
            tickprefix='$'
        ),
        template='plotly_dark',
        height=500,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_dividend_growth(annual_div_df, ticker):
    """Crea análisis de crecimiento de dividendos"""
    if annual_div_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            '<b>Dividendos Anuales</b>',
            '<b>Crecimiento Interanual (%)</b>'
        ),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Dividendos anuales con área de relleno
    fig.add_trace(
        go.Scatter(
            x=annual_div_df['year'],
            y=annual_div_df['annual_dividend'],
            mode='lines+markers',
            name='Dividendo Anual',
            line=dict(color='#00ff88', width=4),
            marker=dict(
                size=12,
                color='#00ff88',
                line=dict(color='white', width=2),
                symbol='circle'
            ),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.15)',
            hovertemplate='<b>Año:</b> %{x}<br><b>Dividendo:</b> $%{y:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # Crecimiento con colores condicionales
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
                hovertemplate='<b>Año:</b> %{x}<br><b>Crecimiento:</b> %{y:.1f}%<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Actualizar ejes
    fig.update_xaxes(
        gridcolor='rgba(255, 255, 255, 0.1)',
        showgrid=True,
        zeroline=False,
        row=1, col=1
    )
    fig.update_xaxes(
        title_text="Año",
        gridcolor='rgba(255, 255, 255, 0.1)',
        showgrid=False,
        zeroline=False,
        row=2, col=1
    )
    
    fig.update_yaxes(
        title_text="Dividendo (USD)",
        gridcolor='rgba(255, 255, 255, 0.1)',
        showgrid=True,
        zeroline=False,
        tickprefix='$',
        row=1, col=1
    )
    fig.update_yaxes(
        title_text="Crecimiento (%)",
        gridcolor='rgba(255, 255, 255, 0.1)',
        showgrid=True,
        zeroline=True,
        zerolinecolor='rgba(255, 255, 255, 0.3)',
        zerolinewidth=2,
        ticksuffix='%',
        row=2, col=1
    )
    
    fig.update_layout(
        height=650,
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        hovermode='x',
        title=dict(
            text=f'<b>{ticker} - Análisis de Crecimiento de Dividendos</b>',
            font=dict(size=20, color='white'),
            x=0.5,
            xanchor='center'
        )
    )
    
    return fig


def main():
    # Encabezado
    st.title("💎 Geraldine Weiss - Análisis de Dividendos")
    st.caption("Plataforma Profesional de Valoración por Dividendos y Estrategia de Inversión")
    
    # Barra lateral
    with st.sidebar:
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
        st.caption("KO · JNJ · PG · MMM · CAT · XOM · CVX · T")
    
    # Contenido principal
    if analyze_button and ticker:
        with st.spinner('🔄 Analizando datos del mercado...'):
            analyzer = GeraldineWeissAnalyzer(ticker.upper(), years)
            
            price_data = analyzer.fetch_price_data()
            dividend_data = analyzer.fetch_dividend_data()
            
            if price_data is None or price_data.empty:
                st.error("❌ No se pudieron obtener datos de precio. Verifica el ticker.")
                return
            
            if dividend_data.empty:
                st.error("❌ No hay datos de dividendos. Esta estrategia requiere acciones que paguen dividendos.")
                return
            
            annual_dividends = analyzer.calculate_annual_dividends(dividend_data)
            analysis_df = analyzer.calculate_valuation_bands(price_data, annual_dividends)
            
            if analysis_df is None or analysis_df.empty:
                st.error("❌ Datos insuficientes para calcular las bandas de valoración.")
                return
            
            signal, description = analyzer.get_current_signal(analysis_df)
            
            # Mensaje de éxito
            st.success(f"✅ Análisis completado para **{ticker.upper()}** con {len(dividend_data)} pagos de dividendos")
            
            # Señal principal
            signal_colors = {
                "COMPRA FUERTE": "#00ff88",
                "COMPRA": "#51cf66",
                "MANTENER": "#ffd93d",
                "VENTA": "#ff8787",
                "VENTA FUERTE": "#ff6b6b"
            }
            
            st.markdown(
                f"""<div class='big-signal' style='border-color: {signal_colors.get(signal, "#ffffff")}; color: {signal_colors.get(signal, "#ffffff")}'>
                {signal}<br>
                <div style='font-size: 18px; margin-top: 10px;'>{description}</div>
                </div>""",
                unsafe_allow_html=True
            )
            
            # Métricas clave
            st.subheader("📊 Métricas Clave")
            
            latest = analysis_df.iloc[-1]
            current_price = latest['Close']
            current_yield = latest['div_yield'] * 100
            undervalued_price = latest['undervalued']
            overvalued_price = latest['overvalued']
            upside_to_undervalued = ((undervalued_price/current_price - 1) * 100)
            upside_to_overvalued = ((overvalued_price/current_price - 1) * 100)
            
            col1, col2, col3, col4 = st.columns(4)
            
            col1.metric(
                "💵 Precio Actual",
                f"${current_price:.2f}",
                help="Último precio de cierre"
            )
            col2.metric(
                "📊 Rentabilidad",
                f"{current_yield:.2f}%",
                help="Rentabilidad por dividendo anual"
            )
            col3.metric(
                "🟢 Zona Infravalorada",
                f"${undervalued_price:.2f}",
                delta=f"{upside_to_undervalued:.1f}%",
                delta_color="inverse",
                help="Nivel de precio infravalorado (compra)"
            )
            col4.metric(
                "🔴 Zona Sobrevalorada",
                f"${overvalued_price:.2f}",
                delta=f"{upside_to_overvalued:.1f}%",
                help="Nivel de precio sobrevalorado (venta)"
            )
            
            st.divider()
            
            # Pestañas
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Análisis de Valoración",
                "💰 Historial de Dividendos",
                "📊 Crecimiento",
                "📚 Guía de Estrategia"
            ])
            
            with tab1:
                st.plotly_chart(
                    plot_geraldine_weiss(analysis_df, ticker.upper()),
                    use_container_width=True
                )
                
                st.subheader("🎯 Interpretación del Gráfico")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info("""
                    **🟢 Zona Infravalorada**
                    
                    Cuando el precio se aproxima a la **línea verde**, la acción ofrece alta rentabilidad por dividendo.
                    
                    - **Acción:** Considerar comprar
                    - **Riesgo:** Menor relativo al histórico
                    - **Retorno:** Apreciación + dividendos
                    """)
                
                with col2:
                    st.warning("""
                    **🔴 Zona Sobrevalorada**
                    
                    Cuando el precio se aproxima a la **línea roja**, la acción ofrece baja rentabilidad por dividendo.
                    
                    - **Acción:** Considerar vender
                    - **Riesgo:** Mayor relativo al histórico
                    - **Retorno:** Potencial alcista limitado
                    """)
            
            with tab2:
                fig_div = plot_dividend_history(dividend_data, ticker.upper())
                if fig_div:
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    st.divider()
                    
                    st.subheader("📊 Estadísticas de Dividendos")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_divs = len(dividend_data)
                    avg_div = dividend_data['amount'].mean()
                    latest_div = dividend_data.iloc[0]['amount'] if len(dividend_data) > 0 else 0
                    total_paid = dividend_data['amount'].sum()
                    
                    col1.metric("📋 Pagos Totales", f"{total_divs:,}")
                    col2.metric("📊 Pago Promedio", f"${avg_div:.3f}")
                    col3.metric("🎯 Último Pago", f"${latest_div:.3f}")
                    col4.metric("💵 Total Acumulado", f"${total_paid:.2f}")
                    
                    st.divider()
                    
                    st.subheader("📅 Pagos de Dividendos Recientes")
                    recent = dividend_data.head(12).copy()
                    recent['ex_dividend_date'] = recent['ex_dividend_date'].dt.strftime('%Y-%m-%d')
                    recent.columns = ['📅 Fecha Ex-Dividendo', '💰 Monto']
                    recent['💰 Monto'] = recent['💰 Monto'].apply(lambda x: f"${x:.3f}")
                    st.dataframe(
                        recent,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "📅 Fecha Ex-Dividendo": st.column_config.TextColumn(width="medium"),
                            "💰 Monto": st.column_config.TextColumn(width="medium")
                        }
                    )
            
            with tab3:
                fig_growth = plot_dividend_growth(annual_dividends, ticker.upper())
                if fig_growth:
                    st.plotly_chart(fig_growth, use_container_width=True)
                    
                    if len(annual_dividends) > 1:
                        st.divider()
                        
                        st.subheader("📈 Métricas de Crecimiento")
                        
                        cagr = ((annual_dividends['annual_dividend'].iloc[-1] / 
                                annual_dividends['annual_dividend'].iloc[0]) ** 
                               (1 / (len(annual_dividends) - 1)) - 1) * 100
                        
                        avg_growth = annual_dividends['annual_dividend'].pct_change().mean() * 100
                        years_data = len(annual_dividends)
                        latest_annual = annual_dividends['annual_dividend'].iloc[-1]
                        first_annual = annual_dividends['annual_dividend'].iloc[0]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        col1.metric(
                            "📊 CAGR",
                            f"{cagr:.2f}%",
                            help="Tasa de Crecimiento Anual Compuesta"
                        )
                        col2.metric(
                            "📈 Crecimiento Promedio",
                            f"{avg_growth:.2f}%",
                            help="Crecimiento anual promedio"
                        )
                        col3.metric(
                            "📅 Período Analizado",
                            f"{years_data} años"
                        )
                        col4.metric(
                            "💰 Último Dividendo Anual",
                            f"${latest_annual:.2f}",
                            delta=f"+${latest_annual - first_annual:.2f} desde inicio"
                        )
            
            with tab4:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    ### 🎓 Resumen de la Estrategia
                    
                    **El Método de Geraldine Weiss** identifica oportunidades mediante patrones históricos de rentabilidad por dividendo.
                    
                    #### Filosofía Central
                    
                    Los precios fluctúan, pero empresas de calidad mantienen dividendos estables, creando patrones predecibles.
                    
                    #### Principios Clave
                    
                    - **Alta Rentabilidad** → Acción infravalorada
                    - **Baja Rentabilidad** → Acción sobrevalorada
                    - **Reversión a la Media** → Las rentabilidades vuelven al promedio
                    
                    #### Implementación
                    
                    1. **Compra**: Cuando precio entra en zona infravalorada
                    2. **Mantener**: Conservar posición cobrando dividendos
                    3. **Venta**: Cuando precio alcanza zona sobrevalorada
                    4. **Repetir**: Reinvertir en nuevas oportunidades
                    """)
                
                with col2:
                    st.markdown("""
                    ### ✅ Candidatos Ideales
                    """)
                    
                    st.success("""
                    **Aristócratas de Dividendos**  
                    25+ años de aumentos consecutivos
                    """)
                    
                    st.success("""
                    **Empresas Blue-Chip**  
                    Líderes del mercado con flujos estables
                    """)
                    
                    st.success("""
                    **Pagadores Consistentes**  
                    Dividendos regulares sin recortes
                    """)
                    
                    st.markdown("### ⚠️ Consideraciones de Riesgo")
                    
                    st.warning("""
                    - Verificar sostenibilidad (payout ratio < 60%)
                    - Monitorear fundamentos continuamente
                    - Diversificar entre sectores
                    - Evitar industrias cíclicas/volátiles
                    - No depender de una sola métrica
                    """)
                
                st.divider()
                
                st.markdown("""
                ### 📊 Rendimiento Histórico
                
                La estrategia de Geraldine Weiss ha entregado históricamente:
                
                - **15-20% de descuento** en entrada al valor razonable
                - **15-20% de prima** en salida al valor razonable
                - **Ingresos consistentes** por dividendos
                - **Apreciación de capital** a largo plazo
                - **Menor volatilidad** vs estrategias de crecimiento
                """)
                
                st.error("""
                **⚠️ Aviso Legal:** Esta herramienta es solo para fines educativos e informativos. 
                No constituye asesoramiento financiero. Consulta con un asesor cualificado antes de invertir.
                El rendimiento pasado no garantiza resultados futuros.
                """)
    
    else:
        # Pantalla de bienvenida
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.info("""
            ### 👋 Bienvenido al Analizador Geraldine Weiss
            
            Esta herramienta profesional implementa la legendaria metodología de valoración por dividendos.
            """)
            
            st.markdown("""
            #### 🎯 Lo Que Ofrece Esta Herramienta
            
            - 📊 Análisis histórico de rentabilidad por dividendo
            - 📈 Bandas dinámicas de valoración (sobrevalorada/infravalorada)
            - 🎯 Señales claras de compra/venta/mantener
            - 💰 Seguimiento completo de pagos de dividendos
            - 📉 Cálculos de tasa de crecimiento y CAGR
            - 🖼️ Visualizaciones interactivas
            
            #### 🚀 Cómo Empezar
            
            1. **Introduce un ticker** en la barra lateral (ej: KO, JNJ, PG)
            2. **Ajusta el período** de análisis (3-10 años)
            3. **Haz clic** en "Analizar Acción"
            4. **Revisa** las bandas de valoración y señales
            """)
            
            st.success("""
            **💡 Tickers Recomendados para Probar:**
            
            KO (Coca-Cola) · JNJ (Johnson & Johnson) · PG (Procter & Gamble)  
            MMM (3M) · CAT (Caterpillar) · XOM (ExxonMobil) · CVX (Chevron)
            """)
    
    # Créditos del autor
    st.markdown("""
    <div class='footer-credit'>
        Desarrollado por <a href='https://bquantfinance.com' target='_blank'>@Gsnchez | bquantfinance.com</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
