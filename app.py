import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

COMPANIES = {
    "Reliance Industries":       "RELIANCE.NS",
    "Tata Consultancy Services": "TCS.NS",
    "HDFC Bank":                 "HDFCBANK.NS",
    "Infosys":                   "INFY.NS",
    "ICICI Bank":                "ICICIBANK.NS",
    "Wipro":                     "WIPRO.NS",
    "State Bank of India":       "SBIN.NS",
    "Tata Motors":               "TATAMOTORS.NS",
    "Coal India":                "COALINDIA.NS",
    "ONGC":                      "ONGC.NS",
    "NTPC":                      "NTPC.NS",
    "Power Grid":                "POWERGRID.NS",
    "Adani Ports":               "ADANIPORTS.NS",
    "Adani Enterprises":         "ADANIENT.NS",
    "Bajaj Finance":             "BAJFINANCE.NS",
    "Asian Paints":              "ASIANPAINT.NS",
    "HCL Technologies":          "HCLTECH.NS",
    "Sun Pharma":                "SUNPHARMA.NS",
    "Maruti Suzuki":             "MARUTI.NS",
    "Larsen & Toubro":           "LT.NS",
    "Axis Bank":                 "AXISBANK.NS",
    "Mahindra & Mahindra":       "M&M.NS",
    "Hindustan Unilever":        "HINDUNILVR.NS",
    "Titan Company":             "TITAN.NS",
    "Tech Mahindra":             "TECHM.NS",
    "NMDC":                      "NMDC.NS",
    "Vedanta":                   "VEDL.NS",
    "JSW Steel":                 "JSWSTEEL.NS",
    "Tata Steel":                "TATASTEEL.NS",
    "Hindalco":                  "HINDALCO.NS",
    "Nifty 50":                  "^NSEI",
    "Sensex (BSE)":              "^BSESN",
    "Nifty Bank":                "^NSEBANK",
    "Nifty IT":                  "^CNXIT",
    "Apple":                     "AAPL",
    "Google (Alphabet)":         "GOOGL",
    "Microsoft":                 "MSFT",
    "Tesla":                     "TSLA",
    "Amazon":                    "AMZN",
    "Meta":                      "META",
    "NVIDIA":                    "NVDA",
    "Netflix":                   "NFLX",
    "JPMorgan Chase":            "JPM",
    "S&P 500":                   "^GSPC",
    "Dow Jones":                 "^DJI",
    "NASDAQ":                    "^IXIC",
}

# TradingView ticker map
TV_MAP = {
    "RELIANCE.NS":"BSE:RELIANCE","TCS.NS":"BSE:TCS","HDFCBANK.NS":"BSE:HDFCBANK",
    "INFY.NS":"BSE:INFY","ICICIBANK.NS":"BSE:ICICIBANK","WIPRO.NS":"BSE:WIPRO",
    "SBIN.NS":"BSE:SBIN","TATAMOTORS.NS":"BSE:TATAMOTORS","COALINDIA.NS":"BSE:COALINDIA",
    "ONGC.NS":"BSE:ONGC","NTPC.NS":"BSE:NTPC","POWERGRID.NS":"BSE:POWERGRID",
    "ADANIPORTS.NS":"BSE:ADANIPORTS","ADANIENT.NS":"BSE:ADANIENT",
    "BAJFINANCE.NS":"BSE:BAJFINANCE","ASIANPAINT.NS":"BSE:ASIANPAINT",
    "HCLTECH.NS":"BSE:HCLTECH","SUNPHARMA.NS":"BSE:SUNPHARMA",
    "MARUTI.NS":"BSE:MARUTI","LT.NS":"BSE:LT","AXISBANK.NS":"BSE:AXISBANK",
    "M&M.NS":"BSE:M_M","HINDUNILVR.NS":"BSE:HINDUNILVR","TITAN.NS":"BSE:TITAN",
    "TECHM.NS":"BSE:TECHM","NMDC.NS":"BSE:NMDC","VEDL.NS":"BSE:VEDL",
    "JSWSTEEL.NS":"BSE:JSWSTEEL","TATASTEEL.NS":"BSE:TATASTEEL","HINDALCO.NS":"BSE:HINDALCO",
    "^NSEI":"NSE:NIFTY","^BSESN":"BSE:SENSEX","^NSEBANK":"NSE:BANKNIFTY","^CNXIT":"NSE:CNXIT",
    "AAPL":"NASDAQ:AAPL","GOOGL":"NASDAQ:GOOGL","MSFT":"NASDAQ:MSFT","TSLA":"NASDAQ:TSLA",
    "AMZN":"NASDAQ:AMZN","META":"NASDAQ:META","NVDA":"NASDAQ:NVDA","NFLX":"NASDAQ:NFLX",
    "JPM":"NYSE:JPM","^GSPC":"SP:SPX","^DJI":"DJ:DJI","^IXIC":"NASDAQ:COMP",
}

st.set_page_config(page_title="StockSense AI", page_icon="🤖", layout="wide", initial_sidebar_state="expanded")

# --- EXNESS THEME ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Roboto+Mono:wght@400;500;700&display=swap');
*, html, body, [class*="css"] { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp { background: #131722; color: #D1D4DC; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1E222D !important;
    border-right: 1px solid #2A2E39 !important;
}
.logo {
    font-family: 'Roboto Mono', monospace; font-size: 1.5rem;
    font-weight: 700; color: #FFFFFF; letter-spacing: 2px; padding: 0.5rem 0;
}
.logo span { color: #FFD000; }
.logo-sub { color: #787B86; font-size: 0.75rem; letter-spacing: 1px; margin-bottom: 10px; }

/* ── Section Titles ── */
.sec-title {
    font-size: 0.75rem; font-weight: 600;
    color: #787B86; text-transform: uppercase; letter-spacing: 1px;
    margin: 1.2rem 0 0.6rem 0; padding-bottom: 4px;
    border-bottom: 1px solid #2A2E39;
}

/* ── Metric Cards ── */
.metric-card {
    background: #1E222D; border: 1px solid #2A2E39; border-radius: 4px;
    padding: 1.2rem 1rem; text-align: center; transition: all 0.2s;
}
.metric-card:hover { border-color: #787B86; }
.metric-label { color: #787B86; font-size: 0.75rem; text-transform: uppercase; font-weight: 500; margin-bottom: 0.4rem; }
.metric-value { font-size: 1.5rem; font-weight: 700; color: #FFFFFF; font-family: 'Roboto Mono', monospace; }
.metric-value.g { color: #22AB94; }
.metric-value.r { color: #F23645; }

/* ── Main Title ── */
.main-title {
    font-size: 2.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px; margin-bottom: 0px;
}
.main-title span { color: #FFD000; }
.main-sub { color: #787B86; font-size: 0.9rem; margin-top: 5px; }

/* ── Buttons (Exness Style) ── */
.stButton > button {
    background: linear-gradient(180deg, #FFD000 0%, #E6BB00 100%) !important;
    color: #000000 !important; border: 1px solid #CCA600 !important; border-radius: 4px !important;
    font-weight: 700 !important; width: 100% !important; padding: 0.8rem !important;
    font-size: 1rem !important; transition: all 0.2s !important;
    box-shadow: 0 4px 15px rgba(255, 208, 0, 0.2) !important;
}
.stButton > button:hover { background: #FFEA00 !important; transform: translateY(-2px); box-shadow: 0 6px 20px rgba(255, 208, 0, 0.3) !important; }

/* ── Inputs ── */
div[data-testid="stTextInput"] input { background: #131722 !important; border: 1px solid #2A2E39 !important; color: #FFFFFF !important; border-radius: 4px !important; font-size: 0.9rem !important; }
div[data-testid="stTextInput"] input:focus { border-color: #FFD000 !important; box-shadow: none !important; }
div[data-testid="stTextInput"] input::placeholder { color: #787B86 !important; }

div[data-testid="stSelectbox"] label, div[data-testid="stTextInput"] label, div[data-testid="stSlider"] label { color: #D1D4DC !important; font-size: 0.8rem !important; font-weight: 500 !important; }

/* ── Info Box ── */
.info-box { background: #2A2E39; border-left: 3px solid #FFD000; border-radius: 4px; padding: 0.8rem; font-size: 0.85rem; color: #D1D4DC; line-height: 1.6; }

/* ── Tabs ── */
button[data-baseweb="tab"] { font-size: 1rem !important; font-weight: 600 !important; color: #787B86 !important; padding-bottom: 12px !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #FFD000 !important; border-bottom: 3px solid #FFD000 !important; }
div[data-testid="stTabs"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════ SIDEBAR ══════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="logo">STOCK<span>SENSE</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="logo-sub">Prediction Engine AI</div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Asset Selection</div>', unsafe_allow_html=True)
    search_query = st.text_input("", placeholder="Search symbol (e.g. TCS)", label_visibility="collapsed", key="search")

    selected_ticker = "RELIANCE.NS"
    selected_name   = "Reliance Industries"

    if search_query:
        q = search_query.lower()
        matches = {n: t for n, t in COMPANIES.items() if q in n.lower() or q in t.lower()}
        if matches:
            chosen = st.selectbox("", list(matches.keys()), label_visibility="collapsed")
            selected_ticker = matches[chosen]
            selected_name   = chosen
            st.markdown(f'<div class="info-box"><b>{selected_name}</b><br>{selected_ticker}</div>', unsafe_allow_html=True)
        else:
            selected_ticker = search_query.upper().strip()
            selected_name   = search_query
            st.markdown(f'<div class="info-box">Custom Symbol<br><b>{selected_ticker}</b></div>', unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Timeframe & Prediction</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: period = st.selectbox("History", ["6mo","1y","2y","5y"], index=1)
    with c2: forecast_days = st.slider("Forecast", 7, 60, 30)

    st.markdown('<div class="sec-title">AI Engine Configuration</div>', unsafe_allow_html=True)
    seq_len = st.slider("Sequence Length", 30, 120, 60)
    use_lstm = False
    epochs = 25

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀 RUN AI PREDICTION")
    st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════ MAIN ═════════════════════════════════════════
st.markdown('<div class="main-title">AI <span>Prediction Terminal</span></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Helper Functions ──────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_data(ticker, period):
    stock = yf.Ticker(ticker)
    df    = stock.history(period=period)
    try:    info = stock.info
    except: info = {}
    return df, info

def add_indicators(df):
    df = df.copy()
    for w in [20,50,100,200]: df[f'MA{w}'] = df['Close'].rolling(w).mean()
    d = df['Close'].diff(); g = d.clip(lower=0).rolling(14).mean(); l = (-d.clip(upper=0)).rolling(14).mean()
    df['RSI']    = 100 - (100/(1+g/l))
    ma = df['Close'].rolling(20).mean(); std = df['Close'].rolling(20).std()
    df['BB_Up']  = ma+2*std; df['BB_Dn'] = ma-2*std
    df['MACD']   = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    df['Signal'] = df['MACD'].ewm(span=9).mean()
    df['Return'] = df['Close'].pct_change()*100
    return df

def metric_card(label, value, color=""):
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value {color}">{value}</div></div>'

def get_tv_symbol(ticker):
    return TV_MAP.get(ticker, ticker)

# ═════════════════════════════ TABS ═══════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🤖 AI Forecast", "📊 Live Chart", "📈 Technicals"])

# ══════════════════════════ TAB 1: AI PREDICTION ══════════════════════════════
with tab1:
    if run_btn or st.session_state.get('prediction_done'):
        if run_btn:
            with st.spinner("Fetching historical data & Initializing ML Models..."):
                try:
                    df, info = fetch_data(selected_ticker, period)
                except Exception as e:
                    st.error(f"Error: {e}"); st.stop()
            if df.empty or len(df) < 60:
                st.error(f"Not enough data for: {selected_ticker}"); st.stop()

            df = df.dropna(); df = add_indicators(df)
            current_price = float(df['Close'].iloc[-1]); prev_price = float(df['Close'].iloc[-2])
            price_change = current_price - prev_price; pct_change = (price_change/prev_price)*100
            high_52w = df['Close'].tail(252).max(); low_52w = df['Close'].tail(252).min()
            company_label = info.get('longName', selected_name)

            st.session_state['prediction_done'] = True
            st.session_state['df']            = df
            st.session_state['company_label'] = company_label
            st.session_state['current_price'] = current_price
            st.session_state['pct_change']    = pct_change
            st.session_state['price_change']  = price_change
            st.session_state['high_52w']      = high_52w
            st.session_state['low_52w']       = low_52w

        df            = st.session_state['df']
        company_label = st.session_state['company_label']
        current_price = st.session_state['current_price']
        pct_change    = st.session_state['pct_change']
        price_change  = st.session_state['price_change']
        high_52w      = st.session_state['high_52w']
        low_52w       = st.session_state['low_52w']

        st.markdown(f'<div class="sec-title">{company_label} ({selected_ticker})</div>', unsafe_allow_html=True)
        c1,c2,c3,c4,c5 = st.columns(5)
        arrow = "▲" if price_change >= 0 else "▼"; clr = "g" if price_change >= 0 else "r"
        c1.markdown(metric_card("Last Price", f"{current_price:.2f}"),              unsafe_allow_html=True)
        c2.markdown(metric_card("Change",     f"{arrow} {abs(pct_change):.2f}%", clr), unsafe_allow_html=True)
        c3.markdown(metric_card("52W High",   f"{high_52w:.2f}", "g"),              unsafe_allow_html=True)
        c4.markdown(metric_card("52W Low",    f"{low_52w:.2f}",  "r"),              unsafe_allow_html=True)
        c5.markdown(metric_card("RSI",        f"{df['RSI'].iloc[-1]:.1f}"),          unsafe_allow_html=True)

        close  = df['Close'].values.reshape(-1,1)
        scaler = MinMaxScaler((0,1)); scaled = scaler.fit_transform(close)
        split  = int(len(scaled)*0.8)

        prog = st.progress(0, text="Initializing models...")
        prog.progress(15, text="Running Linear Regression...")
        X_lr = np.arange(len(close)).reshape(-1,1); lr = LinearRegression()
        lr.fit(X_lr[:split], close[:split]); lr_pred = lr.predict(X_lr[split:])
        lr_rmse = np.sqrt(mean_squared_error(close[split:], lr_pred))
        prog.progress(90, text="Rendering visual output...")

        tdates = df.index[split:]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df.index[:split],y=close[:split].flatten(),name='Training',line=dict(color='#787B86',width=1)))
        fig2.add_trace(go.Scatter(x=tdates,y=close[split:].flatten(),name='Actual',line=dict(color='#FFFFFF',width=2)))
        fig2.add_trace(go.Scatter(x=tdates,y=lr_pred.flatten(),name=f'LinReg RMSE:{lr_rmse:.2f}',line=dict(color='#2962FF',width=2,dash='dash')))

        fig2.update_layout(height=650,plot_bgcolor='#131722',paper_bgcolor='#131722',
            title=dict(text='Machine Learning Price Forecast',font=dict(color='#D1D4DC',family='Inter',size=14)),
            font=dict(color='#787B86',family='Roboto Mono',size=11),
            legend=dict(bgcolor='rgba(30, 34, 45, 0.9)',bordercolor='#2A2E39',borderwidth=1,font=dict(size=11)),
            xaxis=dict(gridcolor='#2A2E39',zerolinecolor='#131722'),
            yaxis=dict(gridcolor='#2A2E39',zerolinecolor='#131722'),
            hovermode='x unified',margin=dict(l=10,r=10,t=50,b=10))
        st.plotly_chart(fig2, use_container_width=True)
        prog.progress(100, text="Process completed.")

        st.markdown('<div class="sec-title">Model Evaluation</div>', unsafe_allow_html=True)
        m1,m2,m3 = st.columns(3)
        m1.markdown(metric_card("LinReg Error",f"{lr_rmse:.2f}"),unsafe_allow_html=True)
        m2.markdown(metric_card("LSTM Error","N/A"),unsafe_allow_html=True)
        m3.markdown(metric_card("Delta","N/A"),unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem;background:#1E222D;border-radius:4px;border:1px solid #2A2E39;margin-top:1rem">
            <h3 style="color:#D1D4DC;font-weight:600;margin-top:1rem">Project Engine Ready</h3>
            <p style="color:#787B86;font-size:0.9rem">Please choose an asset from the sidebar and click<br>
            <strong style="color:#FFD000">🚀 RUN AI PREDICTION</strong> to start</p>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════ TAB 2: TRADINGVIEW LIVE CHART ═════════════════════
with tab2:
    tv_symbol = get_tv_symbol(selected_ticker)
    tv_html = f"""
    <div style="border:1px solid #2A2E39; border-radius:4px; overflow:hidden; background:#131722; height:850px; width:100%;">
      <div class="tradingview-widget-container" style="height:100%; width:100%">
        <div class="tradingview-widget-container__widget" style="height:100%; width:100%"></div>
        <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
        {{
          "autosize": true,
          "symbol": "{tv_symbol}",
          "interval": "D",
          "timezone": "Asia/Kolkata",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "backgroundColor": "#131722",
          "gridColor": "#2A2E39",
          "hide_top_toolbar": false,
          "hide_legend": false,
          "allow_symbol_change": true,
          "save_image": true,
          "calendar": false,
          "studies": ["STD;MACD", "STD;RSI", "STD;Bollinger_Bands"],
          "support_host": "https://www.tradingview.com"
        }}
        </script>
      </div>
    </div>
    """
    components.html(tv_html, height=850)

# ══════════════════════════ TAB 3: TECHNICALS ═════════════════════════════════
with tab3:
    if run_btn or st.session_state.get('prediction_done'):
        df2 = st.session_state.get('df', None)
        if df2 is not None:
            fig3 = make_subplots(rows=4,cols=1,shared_xaxes=True,
                                 row_heights=[0.55,0.15,0.15,0.15],vertical_spacing=0.03)
            fig3.add_trace(go.Scatter(x=df2.index,y=df2['BB_Up'],line=dict(color='rgba(41,98,255,0.3)',width=1),showlegend=False),row=1,col=1)
            fig3.add_trace(go.Scatter(x=df2.index,y=df2['BB_Dn'],fill='tonexty',fillcolor='rgba(41,98,255,0.05)',line=dict(color='rgba(41,98,255,0.3)',width=1),showlegend=False,name='BB'),row=1,col=1)
            fig3.add_trace(go.Candlestick(x=df2.index,open=df2['Open'],high=df2['High'],low=df2['Low'],close=df2['Close'],
                name='Price',increasing=dict(fillcolor='#22AB94',line=dict(color='#22AB94',width=1)),
                decreasing=dict(fillcolor='#F23645',line=dict(color='#F23645',width=1))),row=1,col=1)
            for ma,col in [('MA20','#2962FF'),('MA50','#FFD000'),('MA100','#E040FB'),('MA200','#FF5252')]:
                fig3.add_trace(go.Scatter(x=df2.index,y=df2[ma],name=ma,line=dict(color=col,width=1.2),opacity=0.85),row=1,col=1)
            vc = ['#22AB94' if c>=o else '#F23645' for c,o in zip(df2['Close'],df2['Open'])]
            fig3.add_trace(go.Bar(x=df2.index,y=df2['Volume'],marker_color=vc,opacity=0.6,showlegend=False),row=2,col=1)
            fig3.add_trace(go.Scatter(x=df2.index,y=df2['RSI'],name='RSI',line=dict(color='#E040FB',width=1.5)),row=3,col=1)
            fig3.add_hline(y=70,line_dash='dash',line_color='#787B86',line_width=1,row=3,col=1)
            fig3.add_hline(y=30,line_dash='dash',line_color='#787B86',line_width=1,row=3,col=1)
            mc2 = ['#22AB94' if v>=0 else '#F23645' for v in (df2['MACD']-df2['Signal'])]
            fig3.add_trace(go.Bar(x=df2.index,y=df2['MACD']-df2['Signal'],marker_color=mc2,opacity=0.8,showlegend=False),row=4,col=1)
            fig3.add_trace(go.Scatter(x=df2.index,y=df2['MACD'],name='MACD',line=dict(color='#2962FF',width=1.2)),row=4,col=1)
            fig3.add_trace(go.Scatter(x=df2.index,y=df2['Signal'],name='Signal',line=dict(color='#FFD000',width=1.2)),row=4,col=1)
            fig3.update_layout(height=850,plot_bgcolor='#131722',paper_bgcolor='#131722',
                font=dict(color='#787B86',family='Roboto Mono',size=11),
                xaxis_rangeslider_visible=False,
                legend=dict(bgcolor='rgba(30, 34, 45, 0.95)',bordercolor='#2A2E39',borderwidth=1,
                            font=dict(size=11),orientation='h',yanchor='bottom',y=1.01,xanchor='left',x=0),
                margin=dict(l=10,r=10,t=40,b=10),hovermode='x unified',
                hoverlabel=dict(bgcolor='#1E222D',bordercolor='#FFD000',font=dict(color='#FFFFFF',size=12)))
            for i in range(1,5):
                fig3.update_xaxes(gridcolor='#2A2E39',zerolinecolor='#131722',showspikes=True,spikecolor='#FFD000',spikethickness=1,row=i,col=1)
                fig3.update_yaxes(gridcolor='#2A2E39',zerolinecolor='#131722',row=i,col=1)
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.markdown("""
        <div style="text-align:center;padding:4rem;background:#1E222D;border-radius:4px;border:1px solid #2A2E39;margin-top:1rem">
            <h3 style="color:#D1D4DC;font-weight:600;margin-top:1rem">Technicals Offline</h3>
            <p style="color:#787B86;font-size:0.9rem">Run the Prediction Engine first to generate data.</p>
        </div>""", unsafe_allow_html=True)