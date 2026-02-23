import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import time

st.set_page_config(page_title="NEURAL SENTINEL // AI", layout="wide", page_icon="https://img.icons8.com/nolan/128/artificial-intelligence.png")

def apply_cyber_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
        
        /* Main Background and Font */
        .stApp {
            background: radial-gradient(circle at top, #0a192f 0%, #02060c 100%);
            color: #00f2ff;
            font-family: 'JetBrains Mono', monospace;
        }
        /* Glassmorphism Containers */
        [data-testid="stVerticalBlock"] > div:has(div.stTextArea) {
            background: rgba(0, 242, 255, 0.03);
            border: 1px solid rgba(0, 242, 255, 0.2);
            border-radius: 4px;
            padding: 25px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.5);
        }
        /* Glowing Metric Cards */
        [data-testid="stMetric"] {
            background: rgba(112, 0, 255, 0.05);
            border-left: 3px solid #7000ff;
            padding: 15px;
            border-radius: 4px;
        }
        /* Cyberpunk Button */
        .stButton>button {
            background: transparent;
            color: #00f2ff;
            border: 1px solid #00f2ff;
            border-radius: 0px;
            padding: 10px 20px;
            text-transform: uppercase;
            font-weight: bold;
            letter-spacing: 3px;
            transition: 0.4s;
            position: relative;
            overflow: hidden;
        }
        .stButton>button:hover {
            background: #00f2ff;
            color: #02060c;
            box-shadow: 0 0 30px #00f2ff;
        }
        /* Tabs Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: transparent;
            border: 1px solid rgba(0, 242, 255, 0.2);
            color: #e0e0e0;
            padding: 10px 30px;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(0, 242, 255, 0.1) !important;
            border-color: #00f2ff !important;
        }
        </style>
    """, unsafe_allow_html=True)

apply_cyber_theme()

@st.cache_resource
def load_engine():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("sentiment-analysis", model="", device=device)

engine = load_engine()

with st.sidebar:
    st.image("https://img.icons8.com/nolan/128/artificial-intelligence.png", width=80)
    st.header("CORE_STATUS")
    st.write("● **ENGINE_V:** DISTILBERT_v2")
    st.write("● **CORES:** ACTIVE")
    st.write("● **LATENCY:** 184ms")
    st.progress(92)
    st.markdown("---")
    st.caption("NEURAL SENTINEL // BY AATHISUDHAN A")

st.title("NEURAL SENTINEL // AI")
st.write("---")

tab1, tab2 = st.tabs(["[ SINGLE_NEURAL_SCAN ]", "[ BATCH_DATA_GRID ]"])

with tab1:
    col_l, col_r = st.columns([1.5, 1])

    with col_l:
        input_stream = st.text_area("TARGET_STRING_INPUT", placeholder="Enter data for sentiment extraction...", height=200)
        if st.button("RUN_NEURAL_SCAN"):
            if input_stream:
                with st.status("INITIALIZING NEURAL LAYERS...", expanded=True) as status:
                    st.write("FETCHING WEIGHTS...")
                    time.sleep(0.3)
                    st.write("PARSING SYNTAX...")
                    result = engine(input_stream)[0]
                    status.update(label="SCAN_COMPLETE", state="complete", expanded=False)
                
                with col_r:
                    st.metric("PREDICTED_SENTIMENT", result['label'])
                    st.metric("CONFIDENCE_PROBABILITY", f"{result['score']:.4f}")
                    
                    fig = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = result['score'] * 100,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        gauge = {
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#00f2ff"},
                            'bar': {'color': "#00f2ff"},
                            'bgcolor': "rgba(0,0,0,0)",
                            'borderwidth': 2,
                            'bordercolor': "#7000ff",
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(255, 0, 0, 0.1)'},
                                {'range': [50, 100], 'color': 'rgba(0, 255, 0, 0.1)'}],
                        }))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#00f2ff", 'family': "JetBrains Mono"}, height=250)
                    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.write("TARGET_SOURCE: CSV_UPLOADER")
    file_ptr = st.file_uploader("", type=["csv"])

    if file_ptr:
        df = pd.read_csv(file_ptr)
        if 'text' in df.columns:
            if st.button("EXECUTE_BATCH_PROCESS"):
                with st.spinner("DECRYPTING DATA STREAMS..."):

                    predictions = engine(df['text'].tolist())
                    df['RESULT'] = [p['label'] for p in predictions]
                    df['CONFIDENCE'] = [round(p['score'], 4) for p in predictions]
                    
                    st.dataframe(df.style.set_properties(**{'background-color': '#02060c', 'color': '#00f2ff', 'border-color': '#00f2ff'}), use_container_width=True)
                    
                    fig_anl = px.bar(df, x="RESULT", y="CONFIDENCE", color="RESULT",
                                   color_discrete_map={'POSITIVE': '#00f2ff', 'NEGATIVE': '#7000ff'},
                                   template="plotly_dark", title="NEURAL_DISTRIBUTION_MAP")
                    fig_anl.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig_anl, use_container_width=True)
                    
                    st.download_button("DOWNLOAD_LOG_REPORT", df.to_csv(index=False), "neural_scan_report.csv")
        else:
            st.error("FATAL_ERROR: MISSING 'TEXT' COLUMN IN SOURCE.")
