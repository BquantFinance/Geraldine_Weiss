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
import time

# Page Configuration
st.set_page_config(
    page_title="Geraldine Weiss Dividend Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark Theme CSS
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00ff88;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2839 0%, #151a24 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #00ff88;
        margin: 10px 0;
    }
    .strategy-box {
        background: #1a1f2e;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #2d3748;
        margin: 15px 0;
    }
    .buy-zone {
        color: #00ff88;
        font-weight: bold;
    }
    .sell-zone {
        color: #ff6b6b;
        font-weight: bold;
    }
    .hold-zone {
        color: #ffd93d;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #ffffff;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2839;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3748;
        border-bottom: 2px solid #00ff88;
    }
</style>
""", unsafe_allow_html=True)


class DividendDataFetcher:
    """Fetches dividend data from dividendhistory.org"""
    
    def __init__(self):
        self.base_url = "https://dividendhistory.org/payout"
        self.session = requests.Session()
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.cache = {}
    
    def fetch_dividends(self, ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Fetch dividends via web scraping"""
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
            
        except Exception as e:
            st.warning(f"Could not fetch dividend data: {str(e)}")
            return pd.DataFrame()


class GeraldineWeissAnalyzer:
    """Implements the Geraldine Weiss dividend valuation method"""
    
    def __init__(self, ticker: str, years: int = 6):
        self.ticker = ticker
        self.years = years
        self.dividend_fetcher = DividendDataFetcher()
        
    def fetch_price_data(self):
        """Fetch historical price data"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        try:
            data = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                return None
            data.index = pd.to_datetime(data.index)
            return data
        except Exception as e:
            st.error(f"Error fetching price data: {str(e)}")
            return None
    
    def fetch_dividend_data(self):
        """Fetch dividend data"""
        end_date = datetime.now()
        start_date = end_date - relativedelta(years=self.years)
        
        df = self.dividend_fetcher.fetch_dividends(
            self.ticker, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            # Fallback to yfinance
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
        """Calculate annual dividends by year"""
        if dividend_df.empty:
            return pd.DataFrame()
        
        dividend_df = dividend_df.copy()
        dividend_df['year'] = dividend_df['ex_dividend_date'].dt.year
        
        # Remove outliers (> 2.5 std from mean)
        mean_div = dividend_df['amount'].mean()
        std_div = dividend_df['amount'].std()
        dividend_df = dividend_df[
            abs(dividend_df['amount'] - mean_div) <= (2.5 * std_div)
        ]
        
        annual = dividend_df.groupby('year')['amount'].sum().reset_index()
        annual.columns = ['year', 'annual_dividend']
        
        return annual
    
    def calculate_valuation_bands(self, prices, annual_dividends):
        """Calculate overvalued and undervalued bands"""
        prices = prices.copy()
        prices['year'] = prices.index.year
        
        # Merge with annual dividends
        merged = prices.merge(annual_dividends, on='year', how='inner')
        
        if merged.empty:
            return None
        
        # Calculate dividend yield
        merged['div_yield'] = merged['annual_dividend'] / merged['Close']
        
        # Calculate valuation bands
        max_yield = merged['div_yield'].max()
        min_yield = merged['div_yield'].min()
        
        if max_yield == 0 or min_yield == 0:
            return None
        
        # Undervalued line (high yield = low price)
        merged['undervalued'] = (merged['div_yield'] / max_yield) * merged['Close']
        
        # Overvalued line (low yield = high price)
        merged['overvalued'] = (merged['div_yield'] / min_yield) * merged['Close']
        
        return merged
    
    def get_current_signal(self, analysis_df):
        """Determine current buy/sell signal"""
        if analysis_df is None or analysis_df.empty:
            return "UNKNOWN", "Insufficient data"
        
        latest = analysis_df.iloc[-1]
        price = latest['Close']
        undervalued = latest['undervalued']
        overvalued = latest['overvalued']
        
        # Calculate zones
        range_size = overvalued - undervalued
        lower_buy_zone = undervalued + (range_size * 0.2)
        upper_sell_zone = overvalued - (range_size * 0.2)
        
        if price <= lower_buy_zone:
            return "STRONG BUY", f"Price ${price:.2f} is in the undervalued zone"
        elif price <= undervalued:
            return "BUY", f"Price ${price:.2f} is approaching undervalued level"
        elif price >= upper_sell_zone:
            return "STRONG SELL", f"Price ${price:.2f} is in the overvalued zone"
        elif price >= overvalued:
            return "SELL", f"Price ${price:.2f} is approaching overvalued level"
        else:
            return "HOLD", f"Price ${price:.2f} is in fair value range"


def plot_geraldine_weiss(analysis_df, ticker):
    """Create interactive Geraldine Weiss chart"""
    fig = go.Figure()
    
    # Overvalued band (red)
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['overvalued'],
        name='Overvalued',
        line=dict(color='#ff6b6b', width=2),
        mode='lines'
    ))
    
    # Undervalued band (green)
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['undervalued'],
        name='Undervalued',
        line=dict(color='#00ff88', width=2),
        mode='lines',
        fill='tonexty',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))
    
    # Current price
    fig.add_trace(go.Scatter(
        x=analysis_df.index,
        y=analysis_df['Close'],
        name='Price',
        line=dict(color='#6c5ce7', width=3),
        mode='lines'
    ))
    
    fig.update_layout(
        title=f'{ticker} - Geraldine Weiss Valuation Bands',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        template='plotly_dark',
        hovermode='x unified',
        height=500,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ffffff'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def plot_dividend_history(dividend_df, ticker):
    """Create dividend history visualization"""
    if dividend_df.empty:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dividend_df['ex_dividend_date'],
        y=dividend_df['amount'],
        name='Dividend Amount',
        marker_color='#00ff88',
        hovertemplate='<b>Date:</b> %{x}<br><b>Amount:</b> $%{y:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f'{ticker} - Dividend Payment History',
        xaxis_title='Ex-Dividend Date',
        yaxis_title='Dividend Amount ($)',
        template='plotly_dark',
        height=400,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ffffff')
    )
    
    return fig


def plot_dividend_growth(annual_div_df, ticker):
    """Plot annual dividend growth"""
    if annual_div_df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Annual Dividend Amount', 'Year-over-Year Growth'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Annual dividends
    fig.add_trace(
        go.Scatter(
            x=annual_div_df['year'],
            y=annual_div_df['annual_dividend'],
            mode='lines+markers',
            name='Annual Dividend',
            line=dict(color='#00ff88', width=3),
            marker=dict(size=8)
        ),
        row=1, col=1
    )
    
    # Growth rate
    if len(annual_div_df) > 1:
        growth = annual_div_df['annual_dividend'].pct_change() * 100
        colors = ['#00ff88' if x >= 0 else '#ff6b6b' for x in growth]
        
        fig.add_trace(
            go.Bar(
                x=annual_div_df['year'],
                y=growth,
                name='Growth %',
                marker_color=colors
            ),
            row=2, col=1
        )
    
    fig.update_xaxes(title_text="Year", row=2, col=1)
    fig.update_yaxes(title_text="Dividend ($)", row=1, col=1)
    fig.update_yaxes(title_text="Growth (%)", row=2, col=1)
    
    fig.update_layout(
        height=600,
        template='plotly_dark',
        showlegend=False,
        plot_bgcolor='#0e1117',
        paper_bgcolor='#0e1117',
        font=dict(color='#ffffff')
    )
    
    return fig


# Main App
def main():
    st.title("📊 Geraldine Weiss Dividend Strategy Analyzer")
    st.markdown("**Professional dividend valuation using the time-tested Geraldine Weiss method**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        ticker = st.text_input("Stock Ticker", value="KO", help="Enter stock symbol (e.g., KO, JNJ, PG)")
        years = st.slider("Analysis Period (Years)", 3, 10, 6)
        
        st.markdown("---")
        
        st.markdown("""
        ### 📚 About Geraldine Weiss
        
        Geraldine Weiss pioneered the **dividend yield theory** of stock valuation:
        
        - **Undervalued**: High dividend yield
        - **Overvalued**: Low dividend yield
        - **Buy**: When stock hits lower band
        - **Sell**: When stock hits upper band
        
        This method works best with:
        ✓ Dividend Aristocrats  
        ✓ Stable dividend payers  
        ✓ Blue-chip stocks  
        """)
        
        analyze_button = st.button("🔍 Analyze Stock", type="primary", use_container_width=True)
    
    # Main content
    if analyze_button and ticker:
        with st.spinner(f"Analyzing {ticker.upper()}..."):
            analyzer = GeraldineWeissAnalyzer(ticker.upper(), years)
            
            # Fetch data
            price_data = analyzer.fetch_price_data()
            dividend_data = analyzer.fetch_dividend_data()
            
            if price_data is None or price_data.empty:
                st.error("❌ Could not fetch price data. Please check the ticker symbol.")
                return
            
            if dividend_data.empty:
                st.error("❌ No dividend data found. This method requires dividend-paying stocks.")
                return
            
            # Calculate analysis
            annual_dividends = analyzer.calculate_annual_dividends(dividend_data)
            analysis_df = analyzer.calculate_valuation_bands(price_data, annual_dividends)
            
            if analysis_df is None or analysis_df.empty:
                st.error("❌ Could not calculate valuation bands. Insufficient data.")
                return
            
            signal, description = analyzer.get_current_signal(analysis_df)
            
            # Display results
            st.success(f"✅ Analysis complete for **{ticker.upper()}**")
            
            # Signal banner
            signal_colors = {
                "STRONG BUY": "#00ff88",
                "BUY": "#51cf66",
                "HOLD": "#ffd93d",
                "SELL": "#ff8787",
                "STRONG SELL": "#ff6b6b"
            }
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #1e2839 0%, #151a24 100%); 
                        padding: 30px; border-radius: 15px; text-align: center; 
                        border: 3px solid {signal_colors.get(signal, "#ffffff")}; margin: 20px 0;'>
                <h2 style='color: {signal_colors.get(signal, "#ffffff")}; margin: 0;'>
                    {signal}
                </h2>
                <p style='color: #ffffff; margin: 10px 0 0 0; font-size: 18px;'>
                    {description}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            latest = analysis_df.iloc[-1]
            current_price = latest['Close']
            current_yield = latest['div_yield'] * 100
            undervalued_price = latest['undervalued']
            overvalued_price = latest['overvalued']
            
            with col1:
                st.metric("Current Price", f"${current_price:.2f}")
            with col2:
                st.metric("Dividend Yield", f"{current_yield:.2f}%")
            with col3:
                st.metric("Undervalued Target", f"${undervalued_price:.2f}", 
                         delta=f"{((undervalued_price/current_price - 1) * 100):.1f}%")
            with col4:
                st.metric("Overvalued Target", f"${overvalued_price:.2f}",
                         delta=f"{((overvalued_price/current_price - 1) * 100):.1f}%")
            
            # Tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📈 Valuation Chart", "💰 Dividend History", 
                                               "📊 Dividend Growth", "📋 Strategy Guide"])
            
            with tab1:
                st.plotly_chart(plot_geraldine_weiss(analysis_df, ticker.upper()), 
                               use_container_width=True)
                
                st.markdown("### 🎯 Interpretation")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("""
                    **Green Zone (Undervalued)**
                    - Stock price near lower band
                    - High dividend yield
                    - **Action**: Consider buying
                    - **Risk**: Low relative to historical range
                    """)
                with col2:
                    st.markdown("""
                    **Red Zone (Overvalued)**
                    - Stock price near upper band
                    - Low dividend yield
                    - **Action**: Consider selling
                    - **Risk**: High relative to historical range
                    """)
            
            with tab2:
                fig_div = plot_dividend_history(dividend_data, ticker.upper())
                if fig_div:
                    st.plotly_chart(fig_div, use_container_width=True)
                    
                    # Dividend stats
                    st.markdown("### 📊 Dividend Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_divs = len(dividend_data)
                    avg_div = dividend_data['amount'].mean()
                    latest_div = dividend_data.iloc[0]['amount']
                    total_paid = dividend_data['amount'].sum()
                    
                    with col1:
                        st.metric("Total Payments", total_divs)
                    with col2:
                        st.metric("Average Payment", f"${avg_div:.3f}")
                    with col3:
                        st.metric("Latest Payment", f"${latest_div:.3f}")
                    with col4:
                        st.metric("Total Paid", f"${total_paid:.2f}")
                    
                    # Recent dividends table
                    st.markdown("### 📅 Recent Dividend Payments")
                    recent = dividend_data.head(10).copy()
                    recent['ex_dividend_date'] = recent['ex_dividend_date'].dt.strftime('%Y-%m-%d')
                    st.dataframe(recent[['ex_dividend_date', 'amount']], 
                               use_container_width=True, hide_index=True)
            
            with tab3:
                fig_growth = plot_dividend_growth(annual_dividends, ticker.upper())
                if fig_growth:
                    st.plotly_chart(fig_growth, use_container_width=True)
                    
                    # Growth metrics
                    if len(annual_dividends) > 1:
                        cagr = ((annual_dividends['annual_dividend'].iloc[-1] / 
                                annual_dividends['annual_dividend'].iloc[0]) ** 
                               (1 / (len(annual_dividends) - 1)) - 1) * 100
                        
                        avg_growth = annual_dividends['annual_dividend'].pct_change().mean() * 100
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Dividend CAGR", f"{cagr:.2f}%")
                        with col2:
                            st.metric("Avg Annual Growth", f"{avg_growth:.2f}%")
                        with col3:
                            years_data = len(annual_dividends)
                            st.metric("Years of Data", years_data)
            
            with tab4:
                st.markdown("""
                ### 🎓 Geraldine Weiss Strategy Guide
                
                #### Core Principles
                
                **The Dividend Yield Theory** states that a stock's dividend yield reverts to its historical mean:
                - When yield is **high** (price is low) → Stock is **undervalued**
                - When yield is **low** (price is high) → Stock is **overvalued**
                
                #### How to Use This Tool
                
                1. **Buy Signal**: When stock price approaches or crosses the green (undervalued) line
                2. **Sell Signal**: When stock price approaches or crosses the red (overvalued) line
                3. **Hold**: When price is between the bands (fair value zone)
                
                #### Best Candidates
                
                ✅ **Ideal Stocks**:
                - Dividend Aristocrats (25+ years of increases)
                - Blue-chip companies with stable earnings
                - Consistent dividend payment history
                - Low volatility
                
                ❌ **Avoid**:
                - Non-dividend paying stocks
                - Highly cyclical industries
                - Companies with irregular dividends
                - Recent dividend cuts
                
                #### Risk Management
                
                - **Never** invest based on a single metric
                - Verify company fundamentals
                - Check for dividend sustainability
                - Monitor payout ratio (< 60% is healthy)
                - Diversify across sectors
                
                #### Historical Performance
                
                Geraldine Weiss's method has historically provided:
                - Entry points at 15-20% below fair value
                - Exit points at 15-20% above fair value
                - Consistent income through dividends
                - Long-term capital appreciation
                
                ---
                
                ⚠️ **Disclaimer**: This tool is for educational purposes only. 
                Always conduct thorough research and consult with a financial advisor 
                before making investment decisions.
                """)
    
    else:
        # Welcome screen
        st.markdown("""
        <div class='strategy-box'>
            <h2>👋 Welcome to the Geraldine Weiss Analyzer</h2>
            <p>This professional tool implements the legendary dividend valuation strategy 
            developed by Geraldine Weiss, founder of Investment Quality Trends.</p>
            
            <h3>🎯 What This Tool Does</h3>
            <ul>
                <li>Calculates historical dividend yield ranges</li>
                <li>Identifies overvalued and undervalued zones</li>
                <li>Provides clear buy/sell/hold signals</li>
                <li>Tracks dividend payment history and growth</li>
                <li>Visualizes valuation bands in real-time</li>
            </ul>
            
            <h3>🚀 Getting Started</h3>
            <p>Enter a stock ticker in the sidebar and click "Analyze Stock" to begin.</p>
            
            <p><strong>Suggested tickers to try:</strong> KO, JNJ, PG, MMM, CAT, XOM, CVX, T</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample chart
        st.markdown("### 📊 Sample Analysis")
        st.image("https://via.placeholder.com/1200x400/1e2839/00ff88?text=Enter+a+ticker+to+see+live+analysis", 
                use_container_width=True)


if __name__ == "__main__":
    main()
