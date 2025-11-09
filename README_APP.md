# 🔊 Noise & Mental Health Analytics - Streamlit App

A unified, production-ready Streamlit application that combines advanced analytics, beautiful visualizations, and an intuitive user interface for analyzing the relationship between environmental noise and mental health outcomes.

## ✨ Features

### 🎨 Stunning Landing Page
- Modern, glassmorphism design with smooth animations
- Gradient backgrounds and professional typography
- Feature cards showcasing key capabilities
- Responsive and visually appealing layout

### 📊 Comprehensive Analytics Dashboard
- **Data Upload**: Upload CSV files or use existing datasets
- **Advanced Analysis**: 
  - Feature engineering with interaction terms
  - Multiple ML models (Random Forest, XGBoost, LightGBM, CatBoost, Neural Networks)
  - Hyperparameter optimization
  - Auto-scaling and missing value imputation
- **Interactive Visualizations**:
  - Correlation heatmaps
  - Feature importance charts
  - Model performance comparisons
  - Interactive Plotly dashboards
- **Results Management**:
  - View analysis reports (Markdown and JSON)
  - Download trained models
  - Export visualizations
  - Download analysis reports

## 🚀 Quick Start

### Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Run the App**:
```bash
streamlit run app.py
```

The app will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
.
├── app.py                          # Main Streamlit application
├── Noise_mental_health_final.py   # Analysis modules
├── noise_mental_health_analytics2.py  # Full analytics pipeline
├── requirements.txt                # Python dependencies
├── data/                          # Data directory
├── models/                        # Saved models
├── results/                       # Analysis results and visualizations
└── logs/                          # Log files
```

## 🎯 Usage Guide

### 1. Landing Page
- Start at the beautiful landing page
- Explore feature cards to understand capabilities
- Click "Get Started" to navigate to the dashboard

### 2. Upload Data
- Use the sidebar to upload a CSV file
- Or select an existing dataset from the `data/` folder
- Ensure your data includes:
  - Acoustic features (SPL, spectral features, MFCC, etc.)
  - Mental health metrics (stress scores, anxiety levels, etc.)
  - Environmental context (time, weather, demographics)

### 3. Run Analysis
- Select analysis mode (Quick Analysis or Full Pipeline)
- Click "Run Analysis" button
- Wait for the analysis to complete (may take several minutes)
- View progress indicators and status updates

### 4. Explore Results
- View performance metrics for all models
- Explore interactive visualizations
- Check feature importance rankings
- Review model comparison charts

### 5. Download Results
- Download trained models (`.pkl` files)
- Export analysis reports (Markdown or JSON)
- Save visualizations (PNG images)
- Download interactive dashboards (HTML)

## 🔧 Configuration

### Analysis Options
- **Auto-optimize models**: Enable hyperparameter tuning
- **Analysis Mode**: 
  - Quick Analysis: Runs in-app using Streamlit
  - Full Pipeline: Executes complete analytics script

### View Options
Toggle visibility of:
- Correlation Heatmap
- Feature Importance
- Model Comparison
- Interactive Dashboards
- Analysis Report

## 📊 Data Format

Your CSV file should include columns such as:

**Acoustic Features:**
- `spl_rms`, `spl_peak`, `spl_percentile_90`
- `spectral_centroid_mean`, `spectral_bandwidth_mean`
- `mfcc_*` (MFCC coefficients)
- `roughness`, `sharpness`, `loudness`
- Frequency band energies

**Mental Health Metrics:**
- `composite_stress_score` (target variable)
- `sleep_quality`, `anxiety_level`
- `mood_rating`, `depression_score`

**Environmental Context:**
- `hour`, `day_of_week`, `month`
- `temperature`, `humidity`, `wind_speed`
- `age`, `gender`, `population_density`

## 🎨 Customization

### Styling
The app uses custom CSS for a modern look. Modify the `load_custom_css()` function in `app.py` to customize:
- Colors and gradients
- Fonts and typography
- Animations and transitions
- Layout and spacing

### Adding Features
To extend functionality:
1. Add new visualization functions
2. Integrate additional ML models
3. Create custom analysis pipelines
4. Add new export formats

## 🐛 Troubleshooting

### Import Errors
If you see import errors:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check that `Noise_mental_health_final.py` is in the same directory
- Verify Python version (3.8+ recommended)

### Analysis Fails
- Check data format and required columns
- Ensure sufficient memory for large datasets
- Review error messages in the expandable error details

### Visualizations Not Showing
- Verify that results exist in the `results/` folder
- Check file permissions
- Ensure Plotly is installed: `pip install plotly`

## 📝 Notes

- The app automatically creates necessary directories (`data/`, `models/`, `results/`, `logs/`)
- Models are saved in pickle format for easy loading
- Analysis results are cached in session state for faster navigation
- Large datasets may take several minutes to process

## 🔐 Security

- File uploads are processed in memory
- No data is sent to external servers
- All processing happens locally
- Models and results are stored locally

## 📄 License

This project is part of the Urban Noise Pollution Impact on Mental Health Analytics system.

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📧 Support

For issues or questions:
- Check the error messages in the app
- Review the logs in the `logs/` directory
- Consult the analysis report for detailed information

---

**Built with ❤️ using Streamlit | Powered by Advanced Machine Learning**

