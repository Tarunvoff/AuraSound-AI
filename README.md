# 🔊 Noise & Mental Health Analytics Platform

A comprehensive, production-ready data science and Streamlit application that analyzes and visualizes the complex relationship between environmental noise pollution and mental health outcomes. This platform combines advanced machine learning models, sophisticated audio processing, and an intuitive user interface for stakeholders to explore data, gain insights, and make data-driven decisions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Technical Architecture](#technical-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Core Components](#core-components)
- [Machine Learning Models](#machine-learning-models)
- [Audio Processing](#audio-processing)
- [Data Specifications](#data-specifications)
- [API & Configuration](#api--configuration)
- [Testing](#testing)
- [Results & Outputs](#results--outputs)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project addresses a critical public health concern: the impact of environmental noise pollution on mental health. By leveraging machine learning, audio signal processing, and statistical analysis, the platform enables researchers, urban planners, health professionals, and policymakers to:

- **Analyze** the relationship between noise levels and mental health metrics
- **Predict** stress and mental health outcomes based on soundscape characteristics
- **Visualize** complex patterns through interactive dashboards
- **Generate** actionable insights for noise mitigation policies
- **Model** various environmental factors affecting mental wellbeing

### Problem Statement

Urban noise pollution is a growing public health concern affecting millions globally. This platform bridges the gap between environmental science and mental health research by providing:
- Evidence-based analysis of noise-health correlations
- Predictive models for stress and mental health risk
- Interactive tools for stakeholder engagement

---

## ✨ Key Features

### 🎨 **Modern User Interface**
- **Landing Page**: Glassmorphic design with gradient backgrounds, smooth animations, and professional typography
- **Responsive Layout**: Works seamlessly across desktop and tablet devices
- **Feature Cards**: Showcase key platform capabilities with icons and descriptions
- **Dark/Light Mode Support**: Adaptive to user preferences

### 📊 **Comprehensive Analytics Dashboard**
- **Data Upload**: Support for CSV files with automatic validation
- **Data Exploration**: Statistical summaries and missing value analysis
- **Advanced Feature Engineering**: 
  - Temporal aggregation (hourly, daily, weekly)
  - Interaction terms between noise and health metrics
  - Polynomial features for non-linear relationships
  - Lag features for time-series analysis
  - Soundscape characteristic extraction

### 🤖 **Multi-Model Machine Learning**
Seven advanced regression models with automatic hyperparameter optimization:
1. **Random Forest** - Ensemble method with bagging
2. **XGBoost** - Gradient boosting with regularization
3. **LightGBM** - Fast gradient boosting framework
4. **CatBoost** - Categorical feature handling
5. **Neural Networks** - Deep learning with custom architectures
6. **Support Vector Regression** - For non-linear patterns
7. **Voting Regressor** - Ensemble combining top performers

Features include:
- Automatic scaling and normalization
- Missing value imputation with multiple strategies
- Hyperparameter tuning via GridSearchCV/RandomizedSearchCV
- Cross-validation with configurable folds
- Feature importance ranking
- Model comparison and selection

### 📈 **Interactive Visualizations**
- **Correlation Heatmaps**: Identify relationships between noise metrics and health outcomes
- **Feature Importance Charts**: Understand which factors drive predictions
- **Performance Comparisons**: Side-by-side model accuracy comparison
- **Scatter Plots**: Explore individual variable relationships
- **Time-Series Analysis**: Track trends over time periods
- **Distribution Analysis**: Understand data patterns
- **Confusion-Style Matrices**: Model bias and accuracy breakdown

All visualizations are built with Plotly for interactivity:
- Hover tooltips for detailed values
- Zoom and pan capabilities
- Download as PNG functionality
- Export as interactive HTML

### 🔊 **Advanced Audio Processing**
- **MFCCs** (Mel-Frequency Cepstral Coefficients): 20-coefficient extraction for audio fingerprinting
- **Chroma Features**: 12-dimensional chroma energy distribution
- **Mel Spectrogram**: 128-bin frequency representation
- **Spectral Features**: Centroid, rolloff, bandwidth calculations
- **Zero Crossing Rate**: Temporal texture analysis
- **Temporal Features**: RMS energy and dynamic range
- **Loudness Metrics**: LUFS (Loudness Units relative to Full Scale) calculation

### 📥 **Results Management**
- **Report Generation**: Markdown and JSON formatted analysis reports
- **Model Export**: Save trained models in pickle format for production deployment
- **Visualization Export**: Download matplotlib and Plotly charts
- **Results Storage**: Persistent storage of analysis in `/results` directory
- **Dashboard Sharing**: Export interactive HTML dashboards

---

## 🏗 Technical Architecture

### **Technology Stack**

#### Backend & Data Processing
- **Python 3.8+**: Core language
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **SciPy**: Statistical functions and signal processing
- **Scikit-learn**: Machine learning algorithms and preprocessing

#### Machine Learning
- **XGBoost**: Gradient boosting framework
- **LightGBM**: Fast GBDT implementation (optional)
- **CatBoost**: Categorical boosting (optional)
- **TensorFlow/Keras**: Deep learning models
- **Scikit-learn Ensemble**: Voting and stacking ensembles

#### Audio Processing
- **Librosa** (v0.8.1): Audio feature extraction
- **SoundFile** (v0.10.2): Audio I/O
- **Scikit-MAAD**: Ecoacoustics analysis
- **SoundscapePy**: Soundscape characterization

#### Frontend & Visualization
- **Streamlit** (v1.28+): Web application framework
- **Plotly** (v5.17+): Interactive visualizations
- **Matplotlib/Seaborn**: Static plots
- **Pillow**: Image processing

#### Geospatial (Optional)
- **GeoPandas**: Spatial data handling
- **Folium**: Interactive mapping
- **Geopy**: Location services

### **Architecture Diagram**

```
┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                  │
│  (Streamlit Web App - Landing, Analysis, Prediction)     │
└──────────────────────────┬──────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼────────┐  ┌─────▼──────────┐
│   DATA LAYER │  │  ANALYSIS LAYER │  │   MODEL LAYER  │
├──────────────┤  ├─────────────────┤  ├────────────────┤
│ • CSV Upload │  │ • Soundscape    │  │ • RF/XGBoost   │
│ • Validation │  │   Analyzer      │  │ • LightGBM     │
│ • Profiles   │  │ • Feature       │  │ • CatBoost     │
│ • EDA        │  │   Engineer      │  │ • NN Models    │
│              │  │ • Metrics       │  │ • Ensembles    │
│              │  │   Calculator    │  │ • Hyperparameter│
│              │  │ • Preprocessing │  │   Optimization │
└──────────────┘  └─────────────────┘  └────────────────┘
        │                  │                      │
        └──────────────────┼──────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼──────┐  ┌────────▼────────┐  ┌─────▼──────────┐
│   AUDIO      │  │  VISUALIZATION  │  │   STORAGE      │
│   PROCESSING │  │   & REPORTING   │  │                │
├──────────────┤  ├─────────────────┤  ├────────────────┤
│ • MFCC       │  │ • Correlations  │  │ • Models/      │
│ • Chroma     │  │ • Feature Imp.  │  │   Pickle       │
│ • Mel-spec   │  │ • Performance   │  │ • Reports/     │
│ • Spectral   │  │   Charts        │  │   JSON & MD    │
│ • RMS Energy │  │ • Time-series   │  │ • HTML Dashbds │
│ • LUFS       │  │ • Distribution  │  │ • CSVs         │
└──────────────┘  └─────────────────┘  └────────────────┘
```

---

## 📁 Project Structure

```
c:\DATA\Datascience-proj/
│
├── 📄 app.py                                 # Main Streamlit application
├── 📄 Noise_mental_health_final.py          # Core analysis engine with ML models
├── 📄 noise_mental_health_analytics.py      # Alternative analysis module
├── 📄 noise_mental_health_analytics2.py     # Extended analysis features
├── 📄 data.py                               # Data loading and processing utilities
├── 📄 generate_fake_report.py               # Test data generation
├── 📄 test.py                               # Unit tests
├── 📄 requirements.txt                      # Python dependencies
│
├── 📂 FRONT/noise-mental-health-app/        # Production-ready Streamlit app
│   ├── 📄 app.py                            # Streamlit entry point
│   ├── 📄 config.py                         # Configuration settings
│   ├── 📄 requirements.txt                  # App-specific dependencies
│   │
│   ├── 📂 assets/
│   │   ├── 📂 config/
│   │   │   ├── config.yaml                  # YAML configuration file
│   │   │   └── model_params.json            # Model hyperparameters
│   │   ├── 📂 styles/
│   │   │   └── style.css                    # Custom CSS styling
│   │
│   ├── 📂 src/
│   │   ├── 📂 components/
│   │   │   ├── landing.py                   # Landing page component
│   │   │   ├── analysis.py                  # Analysis dashboard
│   │   │   ├── visualization.py             # Visualization component
│   │   │   └── prediction.py                # Prediction interface
│   │   │
│   │   ├── 📂 models/
│   │   │   ├── soundscape_analyzer.py       # Audio feature extraction
│   │   │   ├── feature_engineer.py          # Feature engineering pipeline
│   │   │   └── stress_predictor.py          # ML model predictions
│   │   │
│   │   ├── 📂 utils/
│   │   │   ├── data_loader.py               # Data I/O functions
│   │   │   ├── preprocessing.py             # Data preprocessing
│   │   │   └── metrics.py                   # Evaluation metrics
│   │
│   ├── 📂 data/
│   │   ├── 📂 raw/                          # Original unprocessed data
│   │   └── 📂 processed/                    # Cleaned and engineered features
│   │
│   ├── 📂 tests/
│   │   ├── test_data_loader.py              # Tests for data loading
│   │   ├── test_models.py                   # Tests for ML models
│   │   └── test_utils.py                    # Tests for utilities
│
├── 📂 app/                                   # Alternative Streamlit app
│   ├── 📄 app.py
│   ├── 📄 config.py
│   ├── 📄 requirements.txt
│   └── 📄 style.css
│
├── 📂 results/                               # Generated reports and dashboards
│   ├── analysis_report.md                   # Markdown analysis summary
│   ├── analysis_results.json                # Detailed JSON results
│   ├── dashboard_*.html                     # Interactive Plotly dashboards
│   │   ├── dashboard_boxplot.html
│   │   ├── dashboard_correlation.html
│   │   ├── dashboard_performance.html
│   │   ├── dashboard_policy.html
│   │   ├── dashboard_scatter.html
│   │   └── dashboard_timeseries.html
│   └── models/                              # Saved trained models
│       └── *.pickle                         # Serialized ML models
│
└── 📂 __pycache__/                          # Python bytecode cache
```

---

## 🚀 Installation & Setup

### **Prerequisites**

- Python 3.8 or higher
- pip package manager
- 2GB+ free disk space (for models and data)
- 4GB+ RAM recommended

### **Step 1: Clone or Prepare Repository**

```bash
# Navigate to your project directory
cd c:\DATA\Datascience-proj

# Or clone from repository
git clone <repository-url> datascience-proj
cd datascience-proj
```

### **Step 2: Create Virtual Environment (Recommended)**

```bash
# Using venv
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### **Step 3: Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt

# If installing the FRONT app specifically:
cd FRONT/noise-mental-health-app
pip install -r requirements.txt
```

### **Step 4: Install Optional Dependencies (Recommended)**

For best performance with advanced models:

```bash
# GPU support (if available)
pip install tensorflow-gpu

# Advanced gradient boosting libraries
pip install lightgbm catboost

# Full audio processing suite
pip install soundscapy scikit-maad
```

### **Step 5: Verify Installation**

```bash
# Test imports
python -c "import streamlit; import pandas; import numpy; print('✓ All core packages installed')"

# Quick model test
python test.py
```

---

## 📖 Usage

### **Running the Main Streamlit Application**

```bash
# From project root
streamlit run app.py

# From FRONT subdirectory
cd FRONT/noise-mental-health-app
streamlit run app.py
```

The app will open at `http://localhost:8501`

### **Basic Workflow**

#### 1. **Landing Page**
   - Read project overview and objectives
   - View key features and capabilities
   - Navigate to analysis or prediction sections

#### 2. **Data Upload & Exploration**
   - Upload CSV file with noise and health data
   - Auto-validate data format and quality
   - View statistical summaries and profiles
   - Identify missing values and outliers

#### 3. **Feature Engineering**
   - Automatic extraction of acoustic features
   - Generate temporal and lagged features
   - Create interaction terms
   - Apply polynomial transformations
   - Scale and normalize features

#### 4. **Model Training**
   - Select models to train (single or multiple)
   - Configure hyperparameters or use automatic optimization
   - Set cross-validation folds (default: 10)
   - Monitor training progress

#### 5. **Analysis & Visualization**
   - View correlation heatmaps
   - Analyze feature importance
   - Compare model performance
   - Explore time-series trends
   - Download visualizations as PNG/HTML

#### 6. **Predictions**
   - Input noise metrics and environmental factors
   - Generate stress/health outcome predictions
   - View confidence intervals
   - Create scenario analysis (if-then simulations)

#### 7. **Results Export**
   - Download trained models (pickle format)
   - Export analysis reports (Markdown/JSON)
   - Share interactive HTML dashboards
   - Archive results for reproducibility

### **Advanced Usage: Programmatic Access**

```python
from Noise_mental_health_final import (
    EnhancedSoundscapeAnalyzer,
    HighAccuracyFeatureEngineer,
    AdvancedStressPredictionModels,
    config
)

# Initialize components
analyzer = EnhancedSoundscapeAnalyzer()
engineer = HighAccuracyFeatureEngineer()
predictor = AdvancedStressPredictionModels()

# Process data
audio_features = analyzer.extract_features(audio_file)
engineered_features = engineer.engineer_features(df)

# Make predictions
stress_prediction = predictor.predict_stress(engineered_features)
confidence = predictor.model.score(X_test, y_test)
```

---

## 🧠 Core Components

### **1. EnhancedSoundscapeAnalyzer**

Sophisticated audio feature extraction from soundscape recordings.

**Extracted Features:**
- **MFCCs** (Mel-Frequency Cepstral Coefficients, 20 coefficients)
  - Perceptually-motivated representation of audio
  - Mimics human auditory perception
  - Used for acoustic scene classification

- **Chroma Features** (12-bin energy distribution)
  - Represents pitch content
  - Useful for music-related analysis
  - Captures harmonic structure

- **Mel Spectrogram** (128-bin representation)
  - Frequency-domain representation
  - Captures spectral characteristics
  - Foundation for many audio tasks

- **Spectral Features**
  - Spectral centroid: Center frequency
  - Spectral rolloff: Frequency below which 95% of energy concentrates
  - Spectral bandwidth: Width of spectrum
  - Zero crossing rate: Temporal texture measure

- **Loudness Metrics**
  - RMS (Root Mean Square) energy
  - LUFS (Loudness Units relative to Full Scale)
  - Dynamic range

**Configuration:**
```python
Config.SAMPLE_RATE = 44100  # Hz
Config.AUDIO_DURATION = 10  # seconds
Config.N_MFCC = 20
Config.N_CHROMA = 12
Config.N_MEL = 128
```

---

### **2. HighAccuracyFeatureEngineer**

Transforms raw data into predictive features via statistical and domain-driven methods.

**Feature Engineering Pipeline:**

1. **Temporal Features**
   - Hour of day, day of week, month, quarter
   - Seasonal indicators
   - Holiday flags

2. **Lagged Features**
   - Previous hour/day values
   - Rolling statistics (mean, std, min, max)
   - Lag order: 1-7 periods

3. **Interaction Terms**
   - Noise × Humidity
   - Noise × Temperature
   - Traffic × Proximity
   - Combined environmental stressors

4. **Polynomial Features**
   - 2nd-degree polynomials
   - Non-linear relationship capture
   - Feature crossing

5. **Aggregate Statistics**
   - Hourly/daily/weekly aggregations
   - Rolling windows
   - Change-rate calculations

6. **Categorical Encoding**
   - One-hot encoding for location/zone
   - Target encoding for high-cardinality features
   - Binary encoding for noise levels

7. **Normalization**
   - StandardScaler: Zero mean, unit variance
   - RobustScaler: Resistant to outliers
   - MinMaxScaler: [0,1] range

**Configuration:**
```python
Config.SEQUENCE_LENGTH = 24
Config.USE_POLYNOMIAL_FEATURES = True
Config.POLY_DEGREE = 2
```

---

### **3. AdvancedStressPredictionModels**

Ensemble of machine learning models with hyperparameter optimization.

**Model Arsenal:**

| Model | Type | Strengths | Use Case |
|-------|------|----------|----------|
| **Random Forest** | Ensemble (Bagging) | Non-linear, robust, interpretable | General baseline |
| **XGBoost** | Ensemble (Boosting) | High accuracy, feature importance | Primary predictor |
| **LightGBM** | Fast Boosting | Speed, memory efficiency | Large datasets |
| **CatBoost** | Boosting | Categorical features, stability | Mixed data types |
| **Neural Network** | Deep Learning | Complex patterns, non-linearity | Complex relationships |
| **Support Vector Regressor** | Kernel Methods | Non-linear mapping, robust | Smaller samples |
| **Voting Ensemble** | Meta | Combines strengths | Final predictions |

**Hyperparameter Optimization:**

The platform performs automatic grid/random search:

```python
# Random Forest
params: {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# XGBoost
params: {
    'max_depth': [5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.9, 1.0]
}

# Neural Network
Architecture:
  - Input layer: [n_features]
  - Hidden 1: [256 neurons, ReLU, Dropout 0.3]
  - Hidden 2: [128 neurons, ReLU, Dropout 0.3]
  - Hidden 3: [64 neurons, ReLU, Dropout 0.2]
  - Output: [1 neuron, Linear]
  
Optimizer: Adam (lr=0.001)
Loss: MSE + L1/L2 regularization
Callbacks: EarlyStopping, ReduceLROnPlateau
```

**Model Evaluation Metrics:**
- **R² Score**: Proportion of variance explained
- **MAE** (Mean Absolute Error): Average absolute deviation
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **Cross-Validation**: 10-fold evaluation
- **Learning Curves**: Training vs. validation analysis

---

## 🔊 Audio Processing Details

### **Audio Feature Extraction Pipeline**

```
Raw Audio File (WAV/MP3)
    ↓
[Librosa Loader]
    ├→ Resample to 44.1 kHz
    ├→ Duration: 10 seconds
    └→ Mono channel
    ↓
[Parallel Feature Extraction]
    ├→ MFCC (20 coeff) + Δ + ΔΔ = 60 features
    ├→ Chroma (12-bin) + Δ + ΔΔ = 36 features
    ├→ Mel-Spectrogram (128-bin, aggregated)
    ├→ Spectral Features (5 metrics)
    ├→ Zero Crossing Rate
    ├→ RMS Energy
    └→ LUFS Loudness
    ↓
[Aggregation]
    ├→ Mean, Std Dev, Min, Max per feature
    └→ Temporal statistics
    ↓
Feature Vector (100+ dimensions)
```

### **Soundscape Characteristics**

The analyzer identifies soundscape type:

- **Urban**: High traffic noise, speech, machinery
- **Natural**: Birdsong, wind, water
- **Industrial**: Machinery, equipment operation
- **Mixed**: Combination of sources
- **Quiet/Silence**: Low ambient noise

**Confidence Score:** 0-1 indicating soundscape classification reliability

---

## 📊 Data Specifications

### **Input Data Format**

**Required Columns (Minimum):**
```python
{
    'timestamp': datetime,           # When measurement taken
    'noise_level_db': float,        # Primary noise metric (dB)
    'stress_score': float,          # Target variable (0-10 scale)
    'location': string,             # Geographic/spatial identifier
}
```

**Optional but Recommended:**
```python
{
    'traffic_count': int,           # Vehicles per hour
    'humidity_pct': float,          # 0-100
    'temperature_c': float,         # Celsius
    'wind_speed_ms': float,         # Meters per second
    'proximity_to_road_m': float,   # Distance in meters
    'population_density': float,    # People per km²
    'green_space_pct': float,       # Percentage
    'construction_activity': bool,  # Yes/No
    'time_of_day': string,          # Morning/Afternoon/Evening/Night
    'day_of_week': string,          # Monday-Sunday
    'event_type': string,           # Concert, festival, etc.
}
```

### **Data Quality Requirements**

- **Size**: Minimum 100 samples, 1000+ recommended
- **Completeness**: <30% missing values suggested
- **Temporal**: Daily or hourly frequency preferred
- **Range Validation**:
  - Noise: 20-120 dB
  - Stress: 0-10 scale
  - Temperature: -50 to +60°C

### **Data Validation Process**

The app automatically:
1. Checks for required columns
2. Validates data types
3. Identifies outliers (beyond 3 std dev)
4. Reports missing value percentages
5. Suggests preprocessing actions
6. Handles missing values via:
   - Forward fill (temporal)
   - Mean imputation (cross-sectional)
   - Model-based imputation (advanced)

---

## ⚙️ API & Configuration

### **Configuration File** (`assets/config/config.yaml`)

```yaml
# Audio Processing
audio:
  sample_rate: 44100
  duration_seconds: 10
  n_mfcc: 20
  n_chroma: 12
  n_mel: 128

# Machine Learning
machine_learning:
  test_size: 0.2
  cv_folds: 10
  random_state: 42
  
# Feature Engineering
features:
  sequence_length: 24
  polynomial_features: true
  poly_degree: 2
  
# Thresholds
thresholds:
  high_noise_db: 70.0
  high_stress_score: 7.0
  
# Paths
paths:
  data_dir: "data"
  models_dir: "models"
  results_dir: "results"
  logs_dir: "logs"
```

### **Model Parameters** (`assets/config/model_params.json`)

```json
{
  "random_forest": {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_split": 5,
    "min_samples_leaf": 2
  },
  "xgboost": {
    "max_depth": 7,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.9
  },
  "lightgbm": {
    "num_leaves": 31,
    "max_depth": 15,
    "learning_rate": 0.05,
    "n_estimators": 200
  }
}
```

### **Programmatic Configuration**

```python
from Noise_mental_health_final import config

# Access settings
sample_rate = config.SAMPLE_RATE          # 44100
cv_folds = config.CV_FOLDS                # 10
data_dir = config.DATA_DIR                # "data"

# Modify if needed
config.TEST_SIZE = 0.25
config.RANDOM_STATE = 123
```

---

## 🧪 Testing

### **Unit Tests**

Run all tests:
```bash
python test.py
```

### **Test Coverage**

**`tests/test_data_loader.py`**
- CSV loading and validation
- Missing value handling
- Data type conversion
- Datetime parsing
- Error handling for invalid files

**`tests/test_models.py`**
- Model initialization
- Training pipeline
- Prediction generation
- Feature importance calculation
- Cross-validation

**`tests/test_utils.py`**
- Preprocessing pipeline
- Scaling transformations
- Feature engineering
- Metrics calculation
- Report generation

### **Integration Tests**

End-to-end workflow:
```python
# 1. Load data
df = load_data("data/raw/sample.csv")

# 2. Preprocess
df_clean = preprocess(df)

# 3. Engineer features
X, y = engineer_features(df_clean)

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 5. Train models
model = train_model(X_train, y_train, model_type='xgboost')

# 6. Evaluate
score, predictions = evaluate(model, X_test, y_test)

# 7. Export
save_model(model, 'models/model_v1.pickle')
```

---

## 📊 Results & Outputs

### **Generated Files**

After running analysis, files are saved to `results/`:

#### **Reports**
- **`analysis_report.md`**: Human-readable summary with markdown formatting
- **`analysis_results.json`**: Machine-readable detailed results

Sample report structure:
```markdown
# Noise and Mental Health Analysis Report
*Generated: 2024-02-26*

## Executive Summary
- Dataset: 500 observations
- Feature Set: 47 engineered features
- Best Model: XGBoost (R² = 0.87)

## Data Overview
- Noise Range: 45-95 dB
- Stress Range: 2-9.5 (0-10 scale)
- Missing Values: 2.3%

## Model Performance
### Training Results
- XGBoost: R²=0.871, MAE=0.45
- Random Forest: R²=0.842, MAE=0.52
- Neural Network: R²=0.856, MAE=0.48

## Feature Importance
1. Noise Level: 0.324
2. Time of Day: 0.156
3. Temperature: 0.089
...

## Recommendations
- Focus on evening noise management (peak stress period)
- Implement green space initiatives
- Establish quiet zones near residential areas
```

#### **Interactive Dashboards**
- **`dashboard_correlation.html`**: Heatmap of variable relationships
- **`dashboard_performance.html`**: Model accuracy metrics
- **`dashboard_scatter.html`**: Feature vs. target scatter plots
- **`dashboard_timeseries.html`**: Temporal trend analysis
- **`dashboard_boxplot.html`**: Distribution comparisons
- **`dashboard_policy.html`**: Policy-relevant metrics

#### **Saved Models**
- **`models/*.pickle`**: Trained ML models for production inference
- Format: Python pickle (compatible across Python versions)
- Includes: Scaler, feature names, hyperparameters

---

## 📚 Data Dictionary

| Variable | Type | Range | Description |
|----------|------|-------|-------------|
| `noise_level_db` | Float | 20-120 | Ambient noise level in decibels |
| `stress_score` | Float | 0-10 | Self-reported stress (target) |
| `traffic_count` | Int | 0+ | Vehicles per hour |
| `temperature_c` | Float | -50 to 60 | Ambient temperature Celsius |
| `humidity_pct` | Float | 0-100 | Relative humidity percentage |
| `wind_speed_ms` | Float | 0+ | Wind speed meters/second |
| `proximity_m` | Float | 0+ | Distance from noise source (m) |
| `hour_of_day` | Int | 0-23 | Hour (temporal feature) |
| `day_of_week` | String | Mon-Sun | Day name |
| `season` | String | Winter/Spring/Summer/Fall | Seasonal indicator |

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### **Getting Started**
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make changes and commit: `git commit -m "Add your feature"`
4. Push and create a Pull Request

### **Code Standards**
- Follow PEP 8 style guide
- Add docstrings to all functions
- Write unit tests for new features
- Update README if adding features

### **Reporting Issues**
- Use GitHub Issues for bug reports
- Include error traces and data samples
- Specify Python version and OS

---

## 📄 License

This project is licensed under the **MIT License**. See LICENSE file for details.

```
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and subject to the following conditions...
```

---

## 🔗 Additional Resources

### **Scientific References**
- Basner et al. (2014). "Auditory and non-auditory effects of noise on health" - Lancet
- WHO Environmental Noise Guidelines (2018)
- Ising & Kruppa (2004). "Health effects caused by noise" - Deutsches Arzteblatt

### **Technical Documentation**
- [Librosa Documentation](https://librosa.org/)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [Streamlit API Reference](https://docs.streamlit.io/)

### **Related Projects**
- [Soundscape Assessment Toolkit](https://www.soundscape-learning.eu/)
- [Openaq Air Quality Data](https://openaq.org/)
- [EPA Noise Regulations](https://www.epa.gov/noise-pollution)

---

## 📞 Support & Contact

For questions or support:
- **GitHub Issues**: Report bugs and feature requests
- **Email**: [Your contact]
- **Documentation**: See `/FRONT/noise-mental-health-app/README.md` for app-specific docs

---

## 🎯 Roadmap

### **Upcoming Features**
- [ ] Real-time noise monitoring integration
- [ ] Geospatial heatmap visualization
- [ ] Mobile app version
- [ ] Advanced scenario planning tools
- [ ] Policy impact simulation
- [ ] Community reporting system
- [ ] API endpoint for programmatic access

### **Under Development**
- [ ] Explainable AI (SHAP) integration
- [ ] Causal inference analysis
- [ ] Time-series forecasting models
- [ ] Automated anomaly detection

---

**Last Updated**: February 26, 2026  
**Version**: 1.0.0  
**Status**: Production Ready

---

*This comprehensive platform bridges environmental science and mental health research, empowering stakeholders with data-driven insights for healthier, quieter communities.*
