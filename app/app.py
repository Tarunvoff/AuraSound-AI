import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from PIL import Image
from datetime import datetime
import json
import sys
import os
from pathlib import Path

# Force dark theme
st.markdown("""
    <style>
        /* Overall dark theme */
        .reportview-container {
            background: #0e1117;
        }
        
        .main {
            color: #fafafa;
            background-color: #0e1117;
        }
        
        /* Dark sidebar */
        .sidebar .sidebar-content {
            background-color: #262730;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #fafafa !important;
        }
        
        /* All text elements */
        p, label, .styled-text {
            color: #fafafa !important;
        }
        
        /* File uploader */
        .uploadedFile {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        
        /* Buttons */
        .stButton>button {
            background-color: #262730;
            color: #fafafa;
            border: 1px solid #4c4c4c;
        }
        
        /* DataFrames */
        .dataframe {
            background-color: #262730 !important;
            color: #fafafa !important;
        }
        
        /* Text inputs */
        .stTextInput>div>div>input {
            background-color: #262730;
            color: #fafafa;
        }
        
        /* Plotly charts */
        .js-plotly-plot {
            background-color: #262730 !important;
        }
        
        .plot-container {
            background-color: #262730 !important;
        }
        
        /* Remove default white background */
        [data-testid="stAppViewContainer"], 
        [data-testid="stHeader"] {
            background: none !important;
        }
        
        [data-testid="stToolbar"] {
            display: none !important;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            background-color: #262730 !important;
        }
        
        [data-testid="stSidebarNav"] {
            background-color: #262730 !important;
        }
        
        [data-testid="stSidebarNavItems"] {
            background-color: #262730 !important;
        }
        
        .stApp {
            background-color: #0e1117 !important;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 10px;
            height: 10px;
        }
        
        ::-webkit-scrollbar-track {
            background: #0e1117;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #4c4c4c;
            border-radius: 5px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Specific element overrides */
        div[data-baseweb="select"] {
            background-color: #262730 !important;
        }
        
        div[role="listbox"] {
            background-color: #262730 !important;
        }
        
        div[role="option"] {
            background-color: #262730 !important;
        }
        
        .element-container {
            background-color: transparent !important;
        }
    </style>
""", unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="Noise Mental Health Analytics",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded")

# Add parent directory to path to import from noise_mental_health_final
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Noise_mental_health_final import (
    EnhancedSoundscapeAnalyzer, 
    HighAccuracyFeatureEngineer, 
    AdvancedStressPredictionModels
)
from app.config import config

# Page config and force dark theme
st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon=config.APP_EMOJI,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    })

# Force dark theme with custom CSS
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0E1117;
            background-image: linear-gradient(to bottom right, #0a0f1c, #111827);
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1E1E1E !important;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }
        section[data-testid="stSidebar"] > div {
            background-color: #1E1E1E !important;
        }
        
        /* Text colors */
        .css-10trblm {
            color: #00f5ff !important;
        }
        .css-1dp5vir {
            color: #eaf2ff !important;
        }
        p, .css-183lzff {
            color: #eaf2ff !important;
        }
        
        /* Buttons */
        .stButton > button {
            background: linear-gradient(90deg, #00f5ff 0%, #007bff 100%) !important;
            color: white !important;
            border: none !important;
            padding: 0.5rem 1rem !important;
            border-radius: 0.5rem !important;
        }
        
        /* File uploader */
        .css-1eqv8xl {
            background-color: #1E1E1E !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Remove default background color */
        div[data-testid="stAppViewContainer"] {
            background: none !important;
        }
        
        /* Headers */
        h1, h2, h3, h4, h5, h6 {
            color: #00f5ff !important;
        }
        
        /* Make sure all text is visible */
        * {
            color: #eaf2ff !important;
        }
        
        /* Add glass effect to containers */
        .stMarkdown div {
            background-color: rgba(30, 30, 30, 0.3) !important;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
    </style>
""", unsafe_allow_html=True)
)

# Set theme config
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: #6C63FF;
            color: white;
        }
        section[data-testid="stSidebar"] .stButton button {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            width: 100%;
        }
        section[data-testid="stSidebar"] .stButton button:hover {
            background-color: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
        }
        [data-testid="stToolbar"] {
            display: none;
        }
        .stDeployButton {
            display: none;
        }
        .st-emotion-cache-1cypcdb {
            background-color: #6C63FF;
            color: white;
        }
        .st-emotion-cache-1hlk0l9 {
            color: white;
        }
        .stAlert {
            background-color: rgba(255, 255, 255, 0.1);
            color: white;
        }
        .stSpinner > div {
            border-top-color: #6C63FF !important;
        }
    </style>
""", unsafe_allow_html=True)

def load_css():
    """Load custom CSS styles"""
    with open('app/style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Main content area */
        .main {
            background: linear-gradient(135deg, #0a0f1c 0%, #111827 100%);
            color: #eaf2ff;
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0a0f1c 0%, #111827 100%);
        }
        
        /* Headers */
        h1 {
            color: #00A6FB;
            font-weight: 600;
            font-size: 2.5rem;
            margin-bottom: 1.5rem;
        }
        
        h2, h3, h4, h5, h6 {
            color: #ffffff;
            font-weight: 500;
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
        
        p {
            color: #E0E0E0;
        }
        
        /* Glass Cards */
        .glass-card {
            border-radius: 1rem;
            padding: 2rem;
            background: rgba(255, 255, 255, 0.06);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            color: #eaf2ff;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .glass-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 2px;
            background: linear-gradient(90deg, #00f5ff, #007bff);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }
        
        .glass-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
            box-shadow: 0 8px 30px rgba(0, 245, 255, 0.1);
        }
        
        .glass-card:hover::before {
            transform: scaleX(1);
        }
        
        /* Buttons */
        .stButton>button {
            border-radius: 0.5rem;
            padding: 0.5rem 2rem;
            font-weight: 500;
            transition: all 0.3s ease;
            background: linear-gradient(90deg, #00A6FB 0%, #0582CA 100%);
            color: white;
            border: none;
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0, 166, 251, 0.2);
        }
        
        /* Metrics */
        .stMetric {
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(0, 166, 251, 0.2);
            color: #ffffff;
        }
        
        /* Plots */
        .plotly-graph {
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        }
        
        /* Sidebar */
        .css-1d391kg {
            background: linear-gradient(180deg, #001B2E 0%, #000C16 100%);
        }
        
        .sidebar .sidebar-content {
            background: rgba(0, 166, 251, 0.05);
        }
        
        /* Custom classes */
        .hero-section {
            text-align: center;
            padding: 4rem 2rem;
            background-size: cover;
            background-position: center;
            color: white;
            border-radius: 1rem;
            margin-bottom: 2rem;
            background-color: rgba(0, 27, 46, 0.7);
        }
        
        .feature-card {
            padding: 1.5rem;
            border-radius: 1rem;
            background: rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
            transition: transform 0.3s ease;
            border: 1px solid rgba(0, 166, 251, 0.1);
            color: #ffffff;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(0, 166, 251, 0.2);
        }
        
        /* DataFrame styling */
        .dataframe {
            background-color: #1B1B1B !important;
            color: #ffffff !important;
        }
        
        .dataframe th {
            background-color: #2C2C2C !important;
            color: #00A6FB !important;
        }
        
        .dataframe td {
            background-color: #1B1B1B !important;
            color: #E0E0E0 !important;
        }
        
        /* File uploader */
        .uploadedFile {
            background-color: #1B1B1B !important;
            color: #ffffff !important;
            border: 1px solid rgba(0, 166, 251, 0.2) !important;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .animate-fade-in {
            animation: fadeIn 1s ease forwards;
        }
        
        /* Additional dark theme elements */
        .stTextInput>div>div>input {
            background-color: #1B1B1B !important;
            color: #ffffff !important;
            border-color: rgba(0, 166, 251, 0.2) !important;
        }
        
        .stSelectbox>div>div>select {
            background-color: #1B1B1B !important;
            color: #ffffff !important;
            border-color: rgba(0, 166, 251, 0.2) !important;
        }
        
        .stMarkdown {
            color: #E0E0E0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

def create_navbar():
    """Create a premium glassmorphic navbar"""
    st.markdown("""
        <nav class="glass-nav">
            <div class="nav-content">
                <div class="nav-logo">
                    🎧 NMA
                </div>
                <div class="nav-links">
                    <a href="#" class="nav-link active">Analysis</a>
                    <a href="#" class="nav-link">Dashboard</a>
                    <a href="#" class="nav-link">Reports</a>
                </div>
            </div>
        </nav>
        <style>
            .glass-nav {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
                background: rgba(10, 15, 28, 0.8);
                backdrop-filter: blur(10px);
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }
            .nav-content {
                max-width: 1200px;
                margin: 0 auto;
                padding: 1rem 2rem;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .nav-logo {
                font-size: 1.5rem;
                font-weight: 700;
                background: linear-gradient(90deg, #00f5ff, #007bff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            .nav-links {
                display: flex;
                gap: 2rem;
            }
            .nav-link {
                color: #eaf2ff;
                text-decoration: none;
                font-weight: 500;
                position: relative;
                padding: 0.5rem 0;
            }
            .nav-link::after {
                content: '';
                position: absolute;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, #00f5ff, #007bff);
                transform: scaleX(0);
                transition: transform 0.3s ease;
            }
            .nav-link:hover::after,
            .nav-link.active::after {
                transform: scaleX(1);
            }
        </style>
    """, unsafe_allow_html=True)

def create_hero_section():
    """Create a visually stunning hero section"""
    hero_bg = "https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&w=2000"
    st.markdown(f"""
        <div class="hero-section animate-fade-in">
            <div class="hero-content">
                <h1 class="hero-title">Noise & Mental Health Analyzer</h1>
                <p class="hero-subtitle">Understanding wellness through data-driven insights</p>
                <div class="hero-buttons">
                    <button class="neon-button primary">
                        <span class="button-text">Login</span>
                        <div class="button-glow"></div>
                    </button>
                    <button class="neon-button secondary">
                        <span class="button-text">Register</span>
                        <div class="button-glow"></div>
                    </button>
                </div>
            </div>
        </div>
        <style>
            .hero-section {
                position: relative;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 4rem 2rem;
                background-image: linear-gradient(rgba(10, 15, 28, 0.8), rgba(17, 24, 39, 0.8)), 
                                url({hero_bg});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            
            .hero-content {
                text-align: center;
                max-width: 800px;
            }
            
            .hero-title {
                font-size: 4.5rem;
                margin-bottom: 1.5rem;
                line-height: 1.2;
                background: linear-gradient(90deg, #00f5ff, #007bff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: glow 2s ease-in-out infinite alternate;
            }
            
            .hero-subtitle {
                font-size: 1.5rem;
                color: rgba(234, 242, 255, 0.8);
                margin-bottom: 3rem;
            }
            
            .hero-buttons {
                display: flex;
                gap: 2rem;
                justify-content: center;
            }
            
            .neon-button {
                position: relative;
                padding: 1rem 2.5rem;
                font-size: 1.1rem;
                font-weight: 500;
                color: #eaf2ff;
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 0.5rem;
                cursor: pointer;
                overflow: hidden;
                transition: all 0.3s ease;
            }
            
            .neon-button.primary {
                background: linear-gradient(90deg, #00f5ff20, #007bff20);
            }
            
            .neon-button.secondary {
                background: linear-gradient(90deg, #b388ff20, #5b46ff20);
            }
            
            .button-glow {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(90deg, #00f5ff, #007bff);
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .neon-button.secondary .button-glow {
                background: linear-gradient(90deg, #b388ff, #5b46ff);
            }
            
            .neon-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 0 20px rgba(0, 245, 255, 0.3);
            }
            
            .neon-button.secondary:hover {
                box-shadow: 0 0 20px rgba(179, 136, 255, 0.3);
            }
            
            .neon-button:hover .button-glow {
                opacity: 0.1;
            }
            
            @keyframes glow {
                from {
                    text-shadow: 0 0 10px rgba(0, 245, 255, 0.2),
                                0 0 20px rgba(0, 245, 255, 0.2),
                                0 0 30px rgba(0, 245, 255, 0.2);
                }
                to {
                    text-shadow: 0 0 20px rgba(0, 245, 255, 0.4),
                                0 0 30px rgba(0, 245, 255, 0.4),
                                0 0 40px rgba(0, 245, 255, 0.4);
                }
            }
        </style>
    """, unsafe_allow_html=True)

def create_feature_cards():
    """Create premium feature cards with animations"""
    st.markdown("""
        <div class="features-grid">
            <div class="feature-card animate-fade-in" data-delay="0">
                <div class="feature-icon">📊</div>
                <h3>Advanced Analytics</h3>
                <p>State-of-the-art machine learning models with hyperparameter optimization.</p>
                <div class="feature-bg"></div>
            </div>
            
            <div class="feature-card animate-fade-in" data-delay="200">
                <div class="feature-icon">🔄</div>
                <h3>Real-time Processing</h3>
                <p>Process environmental noise data in real-time with high precision analysis.</p>
                <div class="feature-bg"></div>
            </div>
            
            <div class="feature-card animate-fade-in" data-delay="400">
                <div class="feature-icon">📈</div>
                <h3>Interactive Insights</h3>
                <p>Dynamic visualizations and dashboards powered by cutting-edge analytics.</p>
                <div class="feature-bg"></div>
            </div>
        </div>
        <style>
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 2rem;
                padding: 4rem 2rem;
            }
            
            .feature-card {
                position: relative;
                padding: 2rem;
                background: rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 1rem;
                overflow: hidden;
                transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            .feature-card:hover {
                transform: translateY(-5px);
                border-color: rgba(0, 245, 255, 0.2);
            }
            
            .feature-icon {
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }
            
            .feature-card h3 {
                font-size: 1.5rem;
                margin-bottom: 1rem;
                background: linear-gradient(90deg, #00f5ff, #007bff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .feature-card p {
                color: rgba(234, 242, 255, 0.8);
                line-height: 1.6;
            }
            
            .feature-bg {
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(45deg, 
                    rgba(0, 245, 255, 0.05) 0%, 
                    rgba(0, 123, 255, 0.05) 100%
                );
                opacity: 0;
                transition: opacity 0.3s ease;
            }
            
            .feature-card:hover .feature-bg {
                opacity: 1;
            }
            
            /* Animation delays */
            .animate-fade-in[data-delay="200"] {
                animation-delay: 200ms;
            }
            
            .animate-fade-in[data-delay="400"] {
                animation-delay: 400ms;
            }
        </style>
    """, unsafe_allow_html=True)

def run_analysis(df):
    """Run the noise-mental health analysis pipeline"""
    with st.spinner("Running analysis..."):
        # Initialize models
        feature_engineer = HighAccuracyFeatureEngineer()
        stress_predictor = AdvancedStressPredictionModels(df)
        
        # Train models and get results
        results = stress_predictor.train_optimized_models()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = config.RESULTS_DIR / f"analysis_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f)
        
        return results, stress_predictor

def create_visualizations(df, results, stress_predictor):
    """Create interactive visualizations"""
    st.subheader("Analysis Results")
    
    # Set dark theme for all plots
    dark_template = dict(
        layout=dict(
            plot_bgcolor='#262730',
            paper_bgcolor='#262730',
            font=dict(color='#fafafa'),
            xaxis=dict(gridcolor='#4c4c4c'),
            yaxis=dict(gridcolor='#4c4c4c')
        )
    )
    
    # Model Performance Comparison
    fig_performance = px.bar(
        pd.DataFrame([
            {'Model': k, 'R² Score': v['r2']} 
            for k, v in results.items()
        ]),
        x='Model',
        y='R² Score',
        title='Model Performance Comparison',
        color='Model',
        template='plotly_dark'
    )
    fig_performance.update_layout(dark_template)
    st.plotly_chart(fig_performance, use_container_width=True)
    
    # Feature Importance
    if hasattr(stress_predictor.best_model, 'feature_importances_'):
        importances = pd.DataFrame({
            'Feature': stress_predictor.feature_names,
            'Importance': stress_predictor.best_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig_importance = px.bar(
            importances.head(15),
            x='Importance',
            y='Feature',
            title='Top 15 Most Important Features',
            orientation='h',
            template='plotly_white'
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # Actual vs Predicted
    y_pred = stress_predictor.best_model.predict(stress_predictor.X_test)
    fig_scatter = px.scatter(
        x=stress_predictor.y_test,
        y=y_pred,
        title='Actual vs Predicted Stress Scores',
        labels={'x': 'Actual', 'y': 'Predicted'},
        template='plotly_white'
    )
    fig_scatter.add_trace(
        go.Scatter(
            x=[stress_predictor.y_test.min(), stress_predictor.y_test.max()],
            y=[stress_predictor.y_test.min(), stress_predictor.y_test.max()],
            mode='lines',
            line=dict(dash='dash', color='red'),
            name='Perfect Prediction'
        )
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

def create_footer():
    """Create a premium footer section"""
    st.markdown("""
        <footer class="premium-footer">
            <div class="footer-content">
                <div class="footer-section">
                    <h4>Noise & Mental Health Analyzer</h4>
                    <p>Understanding wellness through data-driven insights</p>
                </div>
                <div class="footer-section">
                    <h4>Quick Links</h4>
                    <a href="#">Analysis</a>
                    <a href="#">Dashboard</a>
                    <a href="#">Reports</a>
                </div>
                <div class="footer-section">
                    <h4>Connect</h4>
                    <div class="social-icons">
                        <a href="#" class="social-icon">📱</a>
                        <a href="#" class="social-icon">💌</a>
                        <a href="#" class="social-icon">📢</a>
                    </div>
                </div>
            </div>
            <div class="footer-bottom">
                <p>© 2025 Noise & Mental Health Analyzer. All rights reserved.</p>
            </div>
        </footer>
        <style>
            .premium-footer {
                margin-top: 6rem;
                padding: 4rem 2rem 2rem;
                background: rgba(10, 15, 28, 0.9);
                backdrop-filter: blur(10px);
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .footer-content {
                max-width: 1200px;
                margin: 0 auto;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 3rem;
            }
            
            .footer-section h4 {
                font-size: 1.2rem;
                margin-bottom: 1.5rem;
                background: linear-gradient(90deg, #00f5ff, #007bff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .footer-section a {
                display: block;
                color: rgba(234, 242, 255, 0.8);
                text-decoration: none;
                margin-bottom: 0.8rem;
                transition: color 0.3s ease;
            }
            
            .footer-section a:hover {
                color: #00f5ff;
            }
            
            .social-icons {
                display: flex;
                gap: 1rem;
            }
            
            .social-icon {
                font-size: 1.5rem;
                transition: transform 0.3s ease;
            }
            
            .social-icon:hover {
                transform: translateY(-3px);
            }
            
            .footer-bottom {
                margin-top: 3rem;
                padding-top: 2rem;
                text-align: center;
                border-top: 1px solid rgba(255, 255, 255, 0.1);
            }
            
            .footer-bottom p {
                color: rgba(234, 242, 255, 0.6);
                font-size: 0.9rem;
            }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main app function"""
    # Dark theme sidebar styling
    st.markdown("""
        <style>
        [data-testid="stSidebar"] {
            background-color: #262730 !important;
            border-right: 1px solid #0e1117;
        }
        
        [data-testid="stSidebar"] > div:first-child {
            padding: 2rem 1rem;
        }
        
        [data-testid="stSidebar"] h3 {
            color: #fafafa !important;
            font-size: 1.5rem !important;
            font-weight: 500 !important;
            margin-bottom: 2rem !important;
        }
        
        [data-testid="stSidebar"] .stButton button {
            width: 100%;
            background: linear-gradient(90deg, #1e88e5 0%, #005cb2 100%);
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            margin: 0.5rem 0;
            border-radius: 0.3rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }
        
        [data-testid="stSidebar"] .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(30, 136, 229, 0.2);
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar content
    with st.sidebar:
        st.markdown("<h3>Analysis Controls</h3>", unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Dataset (CSV)", type="csv")
        
        st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
        run_analysis_btn = st.button("🚀 Run Analysis")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'results'):
            st.markdown("<div class='sidebar-section'>", unsafe_allow_html=True)
            st.download_button(
                "📥 Download Results",
                data=json.dumps(st.session_state.results, indent=2),
                file_name="analysis_results.json",
                mime="application/json"
            )
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'
    
    if st.session_state.page == 'landing':
        # Landing page
        create_hero_section()
        create_feature_cards()
        
        # Add premium footer
        create_footer()
        
    else:
        # Analysis dashboard
        st.markdown("""
            <div class="dashboard-header">
                <h1>Analysis Dashboard</h1>
                <p class="dashboard-subtitle">Explore your data insights</p>
            </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df.head(), use_container_width=True)
            
            if run_analysis_btn:
                results, stress_predictor = run_analysis(df)
                st.session_state.results = results
                st.session_state.stress_predictor = stress_predictor
                
                # Display results
                create_visualizations(df, results, stress_predictor)
        
        elif hasattr(st.session_state, 'results'):
            create_visualizations(
                None, 
                st.session_state.results,
                st.session_state.stress_predictor
            )
        
        else:
            st.info("Please upload a dataset to begin analysis.")

if __name__ == "__main__":
    main()