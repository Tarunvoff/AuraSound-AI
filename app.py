"""
Unified Streamlit App for Urban Noise Pollution Impact on Mental Health Analytics
Combines landing page, data upload, analysis, visualization, and download functionality
"""

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
import subprocess
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import analysis modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_AVAILABLE = False
try:
    from Noise_mental_health_final import (
        EnhancedSoundscapeAnalyzer, 
        HighAccuracyFeatureEngineer, 
        AdvancedStressPredictionModels,
        config
    )
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    # Don't show warning on initial load - it will show when needed
    ANALYSIS_AVAILABLE = False
except Exception as e:
    # Handle any other errors silently
    ANALYSIS_AVAILABLE = False

# Page Configuration
st.set_page_config(
    page_title="Noise & Mental Health Analytics",
    page_icon="🔊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS & STYLING ====================

def load_custom_css():
    """Load custom CSS for beautiful, modern design"""
    st.markdown("""
    <style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container - Light background */
    .main .block-container {
        background: #ffffff;
        padding: 2rem;
    }
    
    /* Ensure content is visible */
    .stApp {
        background: #f0f2f6;
    }
    
    /* Make sure text is visible */
    .main h1, .main h2, .main h3, .main p {
        color: #1f2937;
    }
    
    /* Override any dark backgrounds */
    .main {
        background: #f0f2f6 !important;
    }
    
    /* Landing Page Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.95) 0%, rgba(118, 75, 162, 0.95) 100%),
                    url('https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-blend-mode: overlay;
        padding: 120px 40px;
        text-align: center;
        border-radius: 20px;
        margin: 20px 0;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.8) 0%, rgba(118, 75, 162, 0.8) 100%);
        z-index: 1;
    }
    
    .hero-content {
        position: relative;
        z-index: 2;
        color: white !important;
    }
    
    .hero-content h1,
    .hero-content p {
        color: white !important;
    }
    
    .hero-title {
        font-size: 4.5rem;
        font-weight: 800;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 10px rgba(0,0,0,0.3);
        animation: fadeInUp 1s ease-out;
    }
    
    .hero-tagline {
        font-size: 1.5rem;
        font-weight: 300;
        margin-bottom: 3rem;
        opacity: 0.95;
        animation: fadeInUp 1.2s ease-out;
    }
    
    .hero-buttons {
        display: flex;
        gap: 20px;
        justify-content: center;
        animation: fadeInUp 1.4s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Glassmorphism Buttons */
    .glass-button {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(10px);
        border: 2px solid rgba(255, 255, 255, 0.3);
        border-radius: 50px;
        padding: 15px 40px;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
    }
    
    .glass-button:hover {
        background: rgba(255, 255, 255, 0.3);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    
    .feature-description {
        color: #7f8c8d;
        line-height: 1.6;
    }
    
    /* Dashboard Styles */
    .dashboard-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 30px;
        border-radius: 20px;
        margin: 20px 0;
    }
    
    /* Metrics Cards */
    .metric-card {
        background: white;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    
    /* Sidebar - use more specific selector */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Streamlit Overrides */
    .stButton>button {
        border-radius: 50px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 40px 20px;
        margin-top: 60px;
        color: #7f8c8d;
        border-top: 1px solid rgba(0,0,0,0.1);
    }
    
    /* Smooth Scroll */
    html {
        scroll-behavior: smooth;
    }
    
    /* Loading Animation */
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .loading-spinner {
        border: 4px solid rgba(102, 126, 234, 0.2);
        border-top: 4px solid #667eea;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== LANDING PAGE ====================

def render_landing_page():
    """Render the stunning landing page"""
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <h1 class="hero-title">🔊 Noise & Mental Health Analytics</h1>
            <p class="hero-tagline">Advanced AI-Powered Analysis of Environmental Noise Impact on Mental Well-being</p>
            <div class="hero-buttons">
                <span style="display: inline-block; margin: 10px;">Use the sidebar to navigate to Dashboard</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards Section
    st.markdown('<a name="features"></a>', unsafe_allow_html=True)
    st.markdown("## ✨ Key Features", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Advanced Analytics</div>
            <div class="feature-description">
                Comprehensive analysis using machine learning models including Random Forest, XGBoost, 
                LightGBM, and Neural Networks with hyperparameter optimization.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">High Accuracy</div>
            <div class="feature-description">
                Auto-optimized models with feature engineering, scaling, and missing value imputation 
                to ensure maximum predictive accuracy even with imprecise data.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Interactive Visualizations</div>
            <div class="feature-description">
                Beautiful, interactive dashboards with correlation heatmaps, feature importance charts, 
                policy impact analysis, and time series visualizations.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    col4, col5, col6 = st.columns(3)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔬</div>
            <div class="feature-title">Soundscape Analysis</div>
            <div class="feature-description">
                Advanced acoustic feature extraction including MFCC, spectral analysis, psychoacoustic 
                features, and frequency band analysis for comprehensive noise characterization.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">💡</div>
            <div class="feature-title">Policy Insights</div>
            <div class="feature-description">
                Generate actionable policy recommendations with cost-effectiveness analysis and 
                expected impact on mental health outcomes.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📥</div>
            <div class="feature-title">Export & Download</div>
            <div class="feature-description">
                Download trained models, analysis reports, visualizations, and prediction outputs 
                in various formats for further analysis and sharing.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Built with ❤️ using Streamlit | Powered by Advanced Machine Learning</p>
        <p style="font-size: 0.9rem; margin-top: 10px;">© 2025 Noise & Mental Health Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

def ensure_directories():
    """Ensure all necessary directories exist"""
    for dir_path in ['data', 'models', 'results', 'logs']:
        Path(dir_path).mkdir(exist_ok=True)

def load_data_from_file(uploaded_file):
    """Load data from uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def run_analysis_pipeline(df, save_to_file=None):
    """Run the complete analysis pipeline"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Feature Engineering
        status_text.text("Step 1/5: Engineering features...")
        progress_bar.progress(20)
        
        feature_engineer = HighAccuracyFeatureEngineer()
        df_engineered = feature_engineer.engineer_features(df)
        
        # Step 2: Model Training
        status_text.text("Step 2/5: Training optimized models (this may take a few minutes)...")
        progress_bar.progress(40)
        
        models = AdvancedStressPredictionModels(df_engineered)
        results = models.train_optimized_models()
        
        # Step 3: Feature Importance
        status_text.text("Step 3/5: Analyzing feature importance...")
        progress_bar.progress(60)
        
        feature_importance = {}
        if hasattr(models, 'best_model') and models.best_model is not None:
            if hasattr(models.best_model, 'feature_importances_'):
                if hasattr(models, 'feature_names'):
                    feature_importance = dict(zip(models.feature_names, models.best_model.feature_importances_))
                else:
                    # Try to get feature names from the model
                    try:
                        n_features = len(models.best_model.feature_importances_)
                        feature_importance = {f'feature_{i}': models.best_model.feature_importances_[i] 
                                            for i in range(n_features)}
                    except:
                        pass
        
        # Step 4: Save data if needed
        status_text.text("Step 4/5: Saving results...")
        progress_bar.progress(80)
        
        if save_to_file:
            try:
                df_engineered.to_csv(save_to_file, index=False)
            except:
                pass
        
        # Step 5: Complete
        status_text.text("Analysis complete!")
        progress_bar.progress(100)
        
        return {
            'models': models,
            'results': results,
            'feature_importance': feature_importance,
            'data': df_engineered
        }
        
    except Exception as e:
        st.error(f"Error during analysis: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None

def execute_full_analysis_script(data_file):
    """Execute the full analysis script (Noise_mental_health_final.py or analytics script)"""
    try:
        # Save uploaded data temporarily
        temp_file = f"data/temp_upload_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Try to execute the analytics script
        script_path = "noise_mental_health_analytics2.py"
        if os.path.exists(script_path):
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        return False, "", "Analysis script not found"
    except subprocess.TimeoutExpired:
        return False, "", "Analysis timed out"
    except Exception as e:
        return False, "", str(e)

def save_model(model, model_name):
    """Save trained model to disk"""
    try:
        model_path = f"models/{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        return model_path
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return None

def load_image(image_path):
    """Load and display image"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        return None
    except Exception as e:
        st.warning(f"Could not load image {image_path}: {e}")
        return None

# ==================== VISUALIZATION FUNCTIONS ====================

def create_correlation_heatmap(df):
    """Create correlation heatmap"""
    try:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Feature Correlation Heatmap",
            width=800,
            height=600,
            xaxis_title="Features",
            yaxis_title="Features"
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create correlation heatmap: {e}")
        return None

def create_feature_importance_chart(feature_importance):
    """Create feature importance visualization"""
    try:
        if not feature_importance:
            return None
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
        features, importances = zip(*sorted_features)
        
        fig = go.Figure(data=go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker=dict(
                color=list(importances),
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title="Top 20 Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600,
            width=800
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create feature importance chart: {e}")
        return None

def create_model_comparison_chart(results):
    """Create model performance comparison chart"""
    try:
        if not results:
            return None
        
        models = list(results.keys())
        r2_scores = [results[m].get('r2', 0) for m in models]
        rmse_scores = [results[m].get('rmse', 0) for m in models]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('R² Score Comparison', 'RMSE Comparison'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig.add_trace(
            go.Bar(x=models, y=r2_scores, name='R² Score', marker_color='#667eea'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='#764ba2'),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Models", row=1, col=1)
        fig.update_xaxes(title_text="Models", row=1, col=2)
        fig.update_yaxes(title_text="R² Score", row=1, col=1)
        fig.update_yaxes(title_text="RMSE", row=1, col=2)
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=500,
            showlegend=False
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create model comparison chart: {e}")
        return None

# ==================== MAIN DASHBOARD ====================

def render_dashboard():
    """Render the main analytics dashboard"""
    st.markdown('<a name="dashboard"></a>', unsafe_allow_html=True)
    st.markdown("# 📊 Analytics Dashboard", unsafe_allow_html=True)
    
    # Sidebar Controls
    with st.sidebar:
        st.markdown("## 🎛️ Controls", unsafe_allow_html=True)
        
        # Data Upload
        st.markdown("### 📤 Upload Data", unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your noise and mental health data in CSV format"
        )
        
        # Use existing data option
        st.markdown("### 📁 Use Existing Data", unsafe_allow_html=True)
        data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
        selected_file = None
        if data_files:
            selected_file = st.selectbox("Select existing dataset", [''] + data_files)
        
        # Analysis Controls
        st.markdown("### ⚙️ Analysis Options", unsafe_allow_html=True)
        analysis_mode = st.radio(
            "Analysis Mode",
            ["Quick Analysis (Streamlit)", "Full Pipeline (Script)"],
            help="Quick Analysis runs in-app. Full Pipeline executes the complete analytics script."
        )
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        auto_optimize = st.checkbox("Auto-optimize models", value=True, help="Enable hyperparameter tuning and feature optimization")
        
        # View Options
        st.markdown("### 👁️ View Options", unsafe_allow_html=True)
        show_heatmap = st.checkbox("Correlation Heatmap", value=True)
        show_feature_importance = st.checkbox("Feature Importance", value=True)
        show_model_comparison = st.checkbox("Model Comparison", value=True)
        show_dashboards = st.checkbox("Interactive Dashboards", value=True)
        show_report = st.checkbox("Analysis Report", value=True)
    
    # Main Content Area
    df = None
    
    # Load data
    if uploaded_file is not None:
        df = load_data_from_file(uploaded_file)
        if df is not None:
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            st.dataframe(df.head(), use_container_width=True)
    elif selected_file:
        try:
            df = pd.read_csv(f"data/{selected_file}")
            st.success(f"✅ Data loaded from {selected_file}! Shape: {df.shape}")
            st.dataframe(df.head(), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
    
    # Run Analysis
    analysis_results = None
    if run_analysis and df is not None:
        if analysis_mode == "Full Pipeline (Script)":
            # Execute full analysis script
            with st.spinner("Running full analysis pipeline... This may take several minutes."):
                # Save data first
                temp_data_file = f"data/temp_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                df.to_csv(temp_data_file, index=False)
                
                # Try to execute the script
                st.info("Executing full analysis pipeline. Please wait...")
                # Note: This would require modifying the script to accept command-line arguments
                # For now, we'll use the in-app analysis
                st.warning("Full pipeline execution requires script modification. Using Quick Analysis instead.")
                analysis_mode = "Quick Analysis (Streamlit)"
        
        if analysis_mode == "Quick Analysis (Streamlit)":
            if not ANALYSIS_AVAILABLE:
                st.error("Analysis modules are not available. Please check your imports.")
            else:
                with st.spinner("Running comprehensive analysis... This may take a few minutes."):
                    save_path = f"data/processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    analysis_results = run_analysis_pipeline(df, save_to_file=save_path)
                
                if analysis_results:
                    st.success("✅ Analysis completed successfully!")
                    
                    # Save models
                    if 'models' in analysis_results and analysis_results['models']:
                        models = analysis_results['models']
                        if hasattr(models, 'best_model') and models.best_model:
                            save_model(models.best_model, 'best_model')
                            st.info("💾 Best model saved to models/best_model.pkl")
                        
                        # Save all models
                        for model_name, model_obj in models.models.items():
                            if model_obj is not None:
                                save_model(model_obj, f"{model_name}_model")
    
    # Display Results
    if analysis_results:
        results = analysis_results.get('results', {})
        feature_importance = analysis_results.get('feature_importance', {})
        df_processed = analysis_results.get('data')
        
        # Metrics Summary
        st.markdown("## 📈 Performance Metrics", unsafe_allow_html=True)
        if results:
            cols = st.columns(min(len(results), 4))
            for idx, (model_name, metrics) in enumerate(list(results.items())[:4]):
                with cols[idx % 4]:
                    st.metric(
                        label=model_name.replace('_', ' ').title(),
                        value=f"{metrics.get('r2', 0):.3f}",
                        delta=f"RMSE: {metrics.get('rmse', 0):.3f}"
                    )
        
        # Visualizations
        if show_heatmap and df_processed is not None:
            st.markdown("## 🔥 Correlation Heatmap", unsafe_allow_html=True)
            fig_heatmap = create_correlation_heatmap(df_processed)
            if fig_heatmap:
                st.plotly_chart(fig_heatmap, use_container_width=True)
        
        if show_feature_importance and feature_importance:
            st.markdown("## 🎯 Feature Importance", unsafe_allow_html=True)
            fig_importance = create_feature_importance_chart(feature_importance)
            if fig_importance:
                st.plotly_chart(fig_importance, use_container_width=True)
        
        if show_model_comparison and results:
            st.markdown("## 📊 Model Comparison", unsafe_allow_html=True)
            fig_comparison = create_model_comparison_chart(results)
            if fig_comparison:
                st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Display Existing Results
    results_dir = Path('results')
    if results_dir.exists():
        # Correlation Heatmap
        if show_heatmap and (results_dir / 'correlation_heatmap.png').exists():
            st.markdown("## 🔥 Correlation Heatmap (Saved)", unsafe_allow_html=True)
            img = load_image('results/correlation_heatmap.png')
            if img:
                st.image(img, use_container_width=True)
        
        # Feature Importance
        if show_feature_importance and (results_dir / 'feature_importance.png').exists():
            st.markdown("## 🎯 Feature Importance (Saved)", unsafe_allow_html=True)
            img = load_image('results/feature_importance.png')
            if img:
                st.image(img, use_container_width=True)
        
        # Model Comparison
        if show_model_comparison and (results_dir / 'model_comparison.png').exists():
            st.markdown("## 📊 Model Comparison (Saved)", unsafe_allow_html=True)
            img = load_image('results/model_comparison.png')
            if img:
                st.image(img, use_container_width=True)
        
        # Interactive Dashboards
        if show_dashboards:
            st.markdown("## 📱 Interactive Dashboards", unsafe_allow_html=True)
            dashboard_files = [
                ('Correlation Dashboard', 'dashboard_correlation.html'),
                ('Time Series Dashboard', 'dashboard_timeseries.html'),
                ('Scatter Dashboard', 'dashboard_scatter.html'),
                ('Boxplot Dashboard', 'dashboard_boxplot.html'),
                ('Performance Dashboard', 'dashboard_performance.html'),
                ('Policy Dashboard', 'dashboard_policy.html')
            ]
            
            for dashboard_name, dashboard_file in dashboard_files:
                dashboard_path = results_dir / dashboard_file
                if dashboard_path.exists():
                    try:
                        with open(dashboard_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                    except UnicodeDecodeError:
                        with open(dashboard_path, 'r', encoding='latin-1') as f:
                            html_content = f.read()
                    st.markdown(f"### {dashboard_name}", unsafe_allow_html=True)
                    st.components.v1.html(html_content, height=600, scrolling=True)
        
        # Analysis Report
        if show_report and (results_dir / 'analysis_report.md').exists():
            st.markdown("## 📄 Analysis Report", unsafe_allow_html=True)
            try:
                # Try UTF-8 first
                with open(results_dir / 'analysis_report.md', 'r', encoding='utf-8') as f:
                    report_content = f.read()
            except UnicodeDecodeError:
                # Fallback to latin-1 or other encodings
                try:
                    with open(results_dir / 'analysis_report.md', 'r', encoding='latin-1') as f:
                        report_content = f.read()
                except:
                    # Last resort: read as binary and decode with errors='ignore'
                    with open(results_dir / 'analysis_report.md', 'rb') as f:
                        report_content = f.read().decode('utf-8', errors='ignore')
            st.markdown(report_content)
            
            # Also try JSON report
            if (results_dir / 'analysis_results.json').exists():
                try:
                    with open(results_dir / 'analysis_results.json', 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    try:
                        with open(results_dir / 'analysis_results.json', 'r', encoding='latin-1') as f:
                            json_data = json.load(f)
                    except:
                        st.warning("Could not load JSON report due to encoding issues.")
                        json_data = None
                if json_data:
                    with st.expander("View Detailed JSON Results"):
                        st.json(json_data)
    
    # Download Section
    st.markdown("## 📥 Downloads", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download Models
        models_dir = Path('models')
        if models_dir.exists():
            model_files = list(models_dir.glob('*.pkl'))
            if model_files:
                st.markdown("### 💾 Trained Models", unsafe_allow_html=True)
                for model_file in model_files:
                    with open(model_file, 'rb') as f:
                        st.download_button(
                            label=f"Download {model_file.name}",
                            data=f.read(),
                            file_name=model_file.name,
                            mime="application/octet-stream"
                        )
    
    with col2:
        # Download Reports
        if (results_dir / 'analysis_report.md').exists():
            st.markdown("### 📄 Reports", unsafe_allow_html=True)
            try:
                with open(results_dir / 'analysis_report.md', 'r', encoding='utf-8') as f:
                    report_data = f.read()
            except UnicodeDecodeError:
                try:
                    with open(results_dir / 'analysis_report.md', 'r', encoding='latin-1') as f:
                        report_data = f.read()
                except:
                    with open(results_dir / 'analysis_report.md', 'rb') as f:
                        report_data = f.read().decode('utf-8', errors='ignore')
            st.download_button(
                label="Download Analysis Report (MD)",
                data=report_data,
                file_name="analysis_report.md",
                mime="text/markdown"
            )
        
        if (results_dir / 'analysis_results.json').exists():
            try:
                with open(results_dir / 'analysis_results.json', 'r', encoding='utf-8') as f:
                    json_data = f.read()
            except UnicodeDecodeError:
                try:
                    with open(results_dir / 'analysis_results.json', 'r', encoding='latin-1') as f:
                        json_data = f.read()
                except:
                    with open(results_dir / 'analysis_results.json', 'rb') as f:
                        json_data = f.read().decode('utf-8', errors='ignore')
            st.download_button(
                label="Download Analysis Results (JSON)",
                data=json_data,
                file_name="analysis_results.json",
                mime="application/json"
            )
    
    with col3:
        # Download Visualizations
        st.markdown("### 🖼️ Visualizations", unsafe_allow_html=True)
        viz_files = [
            'correlation_heatmap.png',
            'feature_importance.png',
            'model_comparison.png',
            'predictions_vs_actual.png',
            'policy_impact_analysis.png',
            'temporal_patterns.png'
        ]
        
        for viz_file in viz_files:
            viz_path = results_dir / viz_file
            if viz_path.exists():
                with open(viz_path, 'rb') as f:
                    st.download_button(
                        label=f"Download {viz_file}",
                        data=f.read(),
                        file_name=viz_file,
                        mime="image/png"
                    )

# ==================== MAIN APP ====================

def main():
    """Main application function"""
    try:
        # Load custom CSS
        load_custom_css()
        
        # Ensure directories exist
        ensure_directories()
        
        # Initialize session state
        if 'show_dashboard' not in st.session_state:
            st.session_state.show_dashboard = False
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["🏠 Landing Page", "📊 Dashboard"],
            index=0 if not st.session_state.show_dashboard else 1
        )
        
        if page == "🏠 Landing Page":
            render_landing_page()
            # Add button to go to dashboard
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Get Started - Go to Dashboard", use_container_width=True, type="primary"):
                    st.session_state.show_dashboard = True
                    st.rerun()
        else:
            render_dashboard()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check the console for more details.")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()

