# Urban Noise Pollution Impact on Mental Health Analytics
# Complete Production-Ready Implementation

import os
import sys
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

# Core Data Science Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, signal
from scipy.optimize import curve_fit
import statsmodels.api as sm

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

# Audio Processing
import librosa
import soundfile as sf

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Neural network models will be disabled.")

# Geospatial
try:
    import geopandas as gpd
    import folium
    from geopy.distance import geodesic
    GEOSPATIAL_AVAILABLE = True
except ImportError:
    GEOSPATIAL_AVAILABLE = False
    print("Geospatial libraries not available. Map features will be disabled.")

# Visualization
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('noise_health_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================

@dataclass
class Config:
    """Configuration settings for the noise-mental health analysis system"""
    
    # Audio Processing
    SAMPLE_RATE: int = 44100
    AUDIO_DURATION: int = 10  # seconds
    N_MFCC: int = 13
    
    # Machine Learning
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    
    # Feature Engineering
    SEQUENCE_LENGTH: int = 24  # hours for LSTM
    
    # Thresholds
    HIGH_NOISE_THRESHOLD: float = 70.0  # dB
    HIGH_STRESS_THRESHOLD: float = 7.0  # 0-10 scale
    
    # File Paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "logs"
    
    def __post_init__(self):
        """Create necessary directories"""
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            Path(dir_path).mkdir(exist_ok=True)

config = Config()

# ==================== DATA MODELS ====================

@dataclass
class AcousticFeatures:
    """Data class for acoustic features"""
    spl_mean: float
    spl_std: float
    spectral_centroid_mean: float
    spectral_centroid_std: float
    mfcc_features: Dict[str, float]
    chroma_mean: float
    chroma_std: float
    zcr_mean: float
    zcr_std: float
    roughness: float
    sharpness: float
    duration: float
    rms_energy: float

@dataclass
class MentalHealthMetrics:
    """Data class for mental health metrics"""
    sleep_quality: float
    anxiety_level: float
    mood_rating: float
    depression_score: float
    composite_stress_score: float

@dataclass
class EnvironmentalContext:
    """Data class for environmental context"""
    timestamp: datetime
    hour: int
    day_of_week: int
    month: int
    season: str
    is_rush_hour: bool
    is_weekend: bool
    temperature: float
    humidity: float
    wind_speed: float
    precipitation: float

# ==================== CORE CLASSES ====================

class SoundscapeAnalyzer:
    """Advanced soundscape analysis with psychoacoustic features"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.bark_scale_bounds = [0, 100, 200, 300, 400, 510, 630, 770, 920, 1080, 
                                 1270, 1480, 1720, 2000, 2320, 2700, 3150, 3700, 
                                 4400, 5300, 6400, 7700, 9500, 12000, 15500]
        logger.info(f"SoundscapeAnalyzer initialized with sample rate: {sample_rate}")
    
    def extract_acoustic_features(self, audio_file: Union[str, np.ndarray]) -> Dict[str, float]:
        """Extract comprehensive acoustic features from audio file or array"""
        try:
            # Load audio
            if isinstance(audio_file, str):
                audio, sr = librosa.load(audio_file, sr=self.sample_rate)
            else:
                audio, sr = audio_file, self.sample_rate
            
            features = {}
            
            # Basic acoustic features
            features.update(self._extract_basic_features(audio, sr))
            
            # Spectral features
            features.update(self._extract_spectral_features(audio, sr))
            
            # MFCC features
            features.update(self._extract_mfcc_features(audio, sr))
            
            # Psychoacoustic features
            features.update(self._extract_psychoacoustic_features(audio, sr))
            
            # Temporal features
            features.update(self._extract_temporal_features(audio, sr))
            
            logger.debug(f"Extracted {len(features)} acoustic features")
            return features
            
        except Exception as e:
            logger.error(f"Error extracting acoustic features: {e}")
            return {}
    
    def _extract_basic_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract basic acoustic features"""
        features = {}
        
        # Sound Pressure Level (SPL)
        rms = np.sqrt(np.mean(audio**2))
        features['spl_mean'] = 20 * np.log10(rms + 1e-10)
        features['spl_std'] = np.std(20 * np.log10(np.abs(audio) + 1e-10))
        
        # RMS Energy
        features['rms_energy'] = rms
        
        # Duration
        features['duration'] = len(audio) / sr
        
        return features
    
    def _extract_spectral_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract spectral features"""
        features = {}
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        return features
    
    def _extract_mfcc_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract MFCC features"""
        features = {}
        
        # MFCCs (perceptual features)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=config.N_MFCC)
        for i in range(config.N_MFCC):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
        
        # Chroma features
        chroma = librosa.feature.chroma(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        return features
    
    def _extract_psychoacoustic_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract psychoacoustic features"""
        features = {}
        
        # Roughness and Sharpness (simplified calculations)
        features['roughness'] = self._calculate_roughness(audio, sr)
        features['sharpness'] = self._calculate_sharpness(audio, sr)
        features['loudness'] = self._calculate_loudness(audio, sr)
        
        return features
    
    def _extract_temporal_features(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Extract temporal features"""
        features = {}
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        features['onset_rate'] = len(onset_frames) / (len(audio) / sr)
        
        # Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = float(tempo)
        except:
            features['tempo'] = 0.0
        
        return features
    
    def _calculate_roughness(self, audio: np.ndarray, sr: int) -> float:
        """Calculate psychoacoustic roughness"""
        try:
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            roughness = np.mean(np.diff(magnitude, axis=0)**2)
            return float(roughness)
        except:
            return 0.0
    
    def _calculate_sharpness(self, audio: np.ndarray, sr: int) -> float:
        """Calculate psychoacoustic sharpness"""
        try:
            freqs = librosa.fft_frequencies(sr=sr)
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            weighted_spectrum = magnitude * (freqs[:, np.newaxis] / 1000)
            sharpness = np.mean(weighted_spectrum)
            return float(sharpness)
        except:
            return 0.0
    
    def _calculate_loudness(self, audio: np.ndarray, sr: int) -> float:
        """Calculate perceived loudness"""
        try:
            freqs, times, spectrogram_data = signal.spectrogram(audio, sr)
            a_weights = self._a_weighting(freqs)
            weighted_spec = spectrogram_data * a_weights[:, np.newaxis]
            loudness = np.mean(np.sum(weighted_spec, axis=0) ** 0.3)
            return float(loudness)
        except:
            return 0.0
    
    def _a_weighting(self, freqs: np.ndarray) -> np.ndarray:
        """Apply A-weighting filter"""
        f = freqs
        f2 = f**2
        A1000 = 7.39705e9
        
        numerator = A1000 * f2**2
        denominator = ((f2 + 20.6**2) * 
                      np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * 
                      (f2 + 12194**2))
        
        return numerator / denominator


class MentalHealthDataProcessor:
    """Process and standardize mental health survey data"""
    
    def __init__(self):
        self.stress_indicators = [
            'sleep_quality', 'anxiety_level', 'mood_rating', 
            'depression_score', 'heart_rate_variability'
        ]
        self.scaler = StandardScaler()
        logger.info("MentalHealthDataProcessor initialized")
    
    def process_survey_data(self, survey_data: pd.DataFrame) -> pd.DataFrame:
        """Process mental health survey data"""
        try:
            df = survey_data.copy()
            
            # Standardize stress indicators (0-10 scale)
            df = self._normalize_indicators(df)
            
            # Create composite stress score
            df = self._create_composite_score(df)
            
            # Add demographic features
            df = self._add_demographic_features(df)
            
            logger.info(f"Processed mental health data: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error processing survey data: {e}")
            return pd.DataFrame()
    
    def _normalize_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize stress indicators to 0-10 scale"""
        for indicator in self.stress_indicators:
            if indicator in df.columns:
                min_val = df[indicator].min()
                max_val = df[indicator].max()
                if max_val > min_val:
                    df[f'{indicator}_normalized'] = ((df[indicator] - min_val) / 
                                                   (max_val - min_val)) * 10
                else:
                    df[f'{indicator}_normalized'] = 5.0  # Default middle value
        return df
    
    def _create_composite_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create composite stress score from multiple indicators"""
        normalized_cols = [col for col in df.columns if 'normalized' in col]
        if normalized_cols:
            df['composite_stress_score'] = df[normalized_cols].mean(axis=1)
        else:
            logger.warning("No normalized indicators found for composite score")
            df['composite_stress_score'] = 5.0  # Default middle value
        return df
    
    def _add_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add demographic and contextual features"""
        # Age groups
        if 'age' in df.columns:
            df['age_group'] = pd.cut(df['age'], 
                                   bins=[0, 25, 35, 50, 65, 100], 
                                   labels=['young', 'young_adult', 'middle', 'older', 'senior'],
                                   include_lowest=True)
        
        # Occupation stress levels
        if 'occupation' in df.columns:
            occupation_stress = {
                'healthcare': 8, 'education': 6, 'finance': 7, 'retail': 5,
                'technology': 6, 'manufacturing': 5, 'government': 4, 'other': 5
            }
            df['occupation_stress'] = df['occupation'].map(occupation_stress).fillna(5)
        
        return df


class EnvironmentalDataIntegrator:
    """Integrate environmental and temporal context data"""
    
    def __init__(self):
        self.weather_features = ['temperature', 'humidity', 'wind_speed', 'precipitation']
        logger.info("EnvironmentalDataIntegrator initialized")
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        try:
            df = df.copy()
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'])
            else:
                logger.warning("No timestamp column found, using current time")
                df['datetime'] = datetime.now()
            
            # Extract temporal features
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
            df['season'] = df['month'].apply(self._get_season)
            
            # Rush hour indicators
            df['is_rush_hour'] = ((df['hour'].between(7, 9)) | 
                                 (df['hour'].between(17, 19))).astype(int)
            
            # Weekend indicator
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
            
            logger.info("Added temporal features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding temporal features: {e}")
            return df
    
    def add_urban_context(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add urban environment features"""
        try:
            df = df.copy()
            
            # Population density
            if 'population' in df.columns and 'area_km2' in df.columns:
                df['population_density'] = df['population'] / df['area_km2']
            
            # Traffic volume categorization
            if 'daily_traffic_count' in df.columns:
                df['traffic_volume_category'] = pd.cut(
                    df['daily_traffic_count'], 
                    bins=3, 
                    labels=['low', 'medium', 'high']
                )
            
            logger.info("Added urban context features")
            return df
            
        except Exception as e:
            logger.error(f"Error adding urban context: {e}")
            return df
    
    def _get_season(self, month: int) -> str:
        """Get season from month"""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'


class NoiseHealthAnalyzer:
    """Analyze correlations and patterns between noise and mental health"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.acoustic_cols = [col for col in df.columns if any(x in col for x in 
                            ['spl', 'spectral', 'mfcc', 'roughness', 'sharpness', 'zcr', 'chroma'])]
        self.health_cols = [col for col in df.columns if any(x in col for x in 
                          ['stress', 'anxiety', 'mood', 'depression'])]
        logger.info(f"NoiseHealthAnalyzer initialized with {len(df)} records")
    
    def correlation_analysis(self) -> pd.DataFrame:
        """Analyze correlations between acoustic features and mental health"""
        try:
            # Correlation matrix
            analysis_cols = self.acoustic_cols + self.health_cols
            corr_matrix = self.df[analysis_cols].corr()
            
            # Focus on acoustic-health correlations
            health_acoustic_corr = corr_matrix.loc[self.health_cols, self.acoustic_cols]
            
            # Create visualization
            plt.figure(figsize=(15, 8))
            sns.heatmap(health_acoustic_corr, annot=True, cmap='RdBu_r', center=0,
                       fmt='.3f', cbar_kws={'label': 'Correlation Coefficient'})
            plt.title('Acoustic Features vs Mental Health Indicators Correlation')
            plt.tight_layout()
            plt.savefig(f'{config.RESULTS_DIR}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Completed correlation analysis")
            return health_acoustic_corr
            
        except Exception as e:
            logger.error(f"Error in correlation analysis: {e}")
            return pd.DataFrame()
    
    def temporal_pattern_analysis(self) -> Dict[str, Any]:
        """Analyze temporal patterns in noise and stress"""
        try:
            results = {}
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Hourly patterns
            if 'hour' in self.df.columns:
                hourly_stress = self.df.groupby('hour')['composite_stress_score'].mean()
                hourly_noise = self.df.groupby('hour')['spl_mean'].mean()
                
                ax1 = axes[0, 0]
                ax1.plot(hourly_stress.index, hourly_stress.values, 'r-', label='Stress', linewidth=2)
                ax1.set_ylabel('Stress Score', color='red')
                ax1.tick_params(axis='y', labelcolor='red')
                
                ax2 = ax1.twinx()
                ax2.plot(hourly_noise.index, hourly_noise.values, 'b-', label='Noise Level', linewidth=2)
                ax2.set_ylabel('Noise Level (dB)', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')
                ax1.set_title('Hourly Stress vs Noise Patterns')
                ax1.set_xlabel('Hour of Day')
                
                results['hourly_correlation'] = hourly_stress.corr(hourly_noise)
            
            # Weekly patterns
            if 'day_of_week' in self.df.columns:
                weekly_stress = self.df.groupby('day_of_week')['composite_stress_score'].mean()
                weekly_noise = self.df.groupby('day_of_week')['spl_mean'].mean()
                
                ax3 = axes[0, 1]
                ax3.plot(weekly_stress.index, weekly_stress.values, 'r-', label='Stress', linewidth=2)
                ax3.set_ylabel('Stress Score', color='red')
                ax3.tick_params(axis='y', labelcolor='red')
                
                ax4 = ax3.twinx()
                ax4.plot(weekly_noise.index, weekly_noise.values, 'b-', label='Noise Level', linewidth=2)
                ax4.set_ylabel('Noise Level (dB)', color='blue')
                ax4.tick_params(axis='y', labelcolor='blue')
                ax3.set_title('Weekly Stress vs Noise Patterns')
                ax3.set_xlabel('Day of Week (0=Monday)')
                
                results['weekly_correlation'] = weekly_stress.corr(weekly_noise)
            
            # Traffic volume vs stress
            if 'traffic_volume_category' in self.df.columns:
                sns.boxplot(data=self.df, x='traffic_volume_category', y='composite_stress_score', ax=axes[1, 0])
                axes[1, 0].set_title('Stress Levels by Traffic Volume')
            
            # Age group analysis
            if 'age_group' in self.df.columns:
                sns.boxplot(data=self.df, x='age_group', y='composite_stress_score', ax=axes[1, 1])
                axes[1, 1].set_title('Stress Levels by Age Group')
                axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{config.RESULTS_DIR}/temporal_patterns.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Completed temporal pattern analysis")
            return results
            
        except Exception as e:
            logger.error(f"Error in temporal pattern analysis: {e}")
            return {}
    
    def statistical_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive statistical analysis"""
        try:
            results = {}
            
            # Pearson correlation test
            if 'spl_mean' in self.df.columns and 'composite_stress_score' in self.df.columns:
                corr_coef, p_value = stats.pearsonr(self.df['spl_mean'], self.df['composite_stress_score'])
                results['correlation_test'] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            
            # Group comparison (high vs low noise exposure)
            median_noise = self.df['spl_mean'].median()
            high_noise_stress = self.df[self.df['spl_mean'] > median_noise]['composite_stress_score']
            low_noise_stress = self.df[self.df['spl_mean'] <= median_noise]['composite_stress_score']
            
            t_stat, t_p = stats.ttest_ind(high_noise_stress, low_noise_stress)
            results['group_comparison'] = {
                't_statistic': t_stat,
                'p_value': t_p,
                'significant': t_p < 0.05,
                'high_noise_mean': high_noise_stress.mean(),
                'low_noise_mean': low_noise_stress.mean(),
                'effect_size': (high_noise_stress.mean() - low_noise_stress.mean()) / 
                              np.sqrt((high_noise_stress.var() + low_noise_stress.var()) / 2)
            }
            
            # Linear regression analysis
            feature_cols = ['spl_mean', 'hour', 'is_weekend']
            available_features = [col for col in feature_cols if col in self.df.columns]
            
            if available_features and 'composite_stress_score' in self.df.columns:
                X = self.df[available_features].fillna(0)
                X = sm.add_constant(X)
                y = self.df['composite_stress_score']
                
                model = sm.OLS(y, X).fit()
                results['regression_analysis'] = {
                    'r_squared': model.rsquared,
                    'f_statistic': model.fvalue,
                    'f_p_value': model.f_pvalue,
                    'coefficients': dict(model.params),
                    'p_values': dict(model.pvalues)
                }
            
            logger.info("Completed statistical analysis")
            return results
            
        except Exception as e:
            logger.error(f"Error in statistical analysis: {e}")
            return {}


class StressPredictionModels:
    """Train and evaluate machine learning models for stress prediction"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.scalers = {}
        self.feature_cols = []
        self.results = {}
        logger.info(f"StressPredictionModels initialized with {len(df)} records")
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features for machine learning"""
        try:
            # Feature selection
            feature_cols = [col for col in self.df.columns if any(x in col for x in 
                           ['spl', 'spectral', 'mfcc', 'roughness', 'sharpness', 'zcr', 'rms',
                            'chroma', 'loudness', 'hour', 'day_of_week', 'is_rush_hour', 
                            'is_weekend', 'population_density', 'occupation_stress', 
                            'temperature', 'humidity', 'wind_speed'])]
            
            # Handle categorical variables
            df_encoded = self.df.copy()
            categorical_cols = ['age_group', 'traffic_volume_category', 'season']
            
            for col in categorical_cols:
                if col in df_encoded.columns:
                    # One-hot encoding
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    feature_cols.extend(dummies.columns.tolist())
            
            # Remove original categorical columns and select final features
            available_features = [col for col in feature_cols if col in df_encoded.columns]
            
            X = df_encoded[available_features].fillna(0)
            y = df_encoded['composite_stress_score']
            
            self.feature_cols = available_features
            logger.info(f"Prepared {len(available_features)} features for modeling")
            
            return X, y, available_features
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series(), []
    
    def train_models(self) -> Dict[str, Dict[str, float]]:
        """Train multiple models for stress prediction"""
        try:
            X, y, feature_cols = self.prepare_features()
            
            if X.empty or y.empty:
                logger.error("No data available for training")
                return {}
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['robust'] = scaler
            
            # Initialize models
            models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=config.RANDOM_STATE,
                    n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, 
                    random_state=config.RANDOM_STATE
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    random_state=config.RANDOM_STATE,
                    verbosity=0
                ),
                'ridge': Ridge(alpha=1.0),
                'svr': SVR(kernel='rbf', C=1.0)
            }
            
            results = {}
            predictions = {}
            
            # Train and evaluate each model
            for name, model in models.items():
                logger.info(f"Training {name} model...")
                
                # Use scaled data for SVR and Ridge, original for tree-based models
                if name in ['svr', 'ridge']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Store model and predictions
                self.models[name] = model
                predictions[name] = y_pred
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                results[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
                
                logger.info(f"{name} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
            
            # Neural network (if TensorFlow is available)
            if TF_AVAILABLE:
                logger.info("Training neural network model...")
                nn_model = self._build_neural_network(X_train_scaled.shape[1])
                
                # Early stopping callback
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
                
                history = nn_model.fit(
                    X_train_scaled, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    callbacks=[early_stopping]
                )
                
                nn_pred = nn_model.predict(X_test_scaled, verbose=0).flatten()
                
                self.models['neural_network'] = nn_model
                predictions['neural_network'] = nn_pred
                
                # NN metrics
                nn_mse = mean_squared_error(y_test, nn_pred)
                nn_rmse = np.sqrt(nn_mse)
                nn_mae = mean_absolute_error(y_test, nn_pred)
                nn_r2 = r2_score(y_test, nn_pred)
                
                results['neural_network'] = {
                    'mse': nn_mse,
                    'rmse': nn_rmse,
                    'mae': nn_mae,
                    'r2': nn_r2
                }
                
                logger.info(f"neural_network - R²: {nn_r2:.4f}, RMSE: {nn_rmse:.4f}, MAE: {nn_mae:.4f}")
            
            # Store results and test data for ensemble
            self.results = results
            self.X_test = X_test
            self.y_test = y_test
            self.predictions = predictions
            
            # Create results visualization
            self._plot_model_comparison(results)
            self._plot_predictions_vs_actual(predictions, y_test)
            
            logger.info("Model training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return {}
    
    def _build_neural_network(self, input_dim: int):
        """Build neural network for stress prediction"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def feature_importance_analysis(self) -> pd.DataFrame:
        """Analyze feature importance using tree-based models"""
        try:
            if 'random_forest' not in self.models:
                logger.warning("Random Forest model not available for feature importance")
                return pd.DataFrame()
            
            # Get feature importance from Random Forest
            rf_importance = self.models['random_forest'].feature_importances_
            importance_df = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': rf_importance
            }).sort_values('importance', ascending=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            sns.barplot(data=top_features, x='importance', y='feature', palette='viridis')
            plt.title('Top 20 Most Important Features for Stress Prediction')
            plt.xlabel('Feature Importance')
            plt.tight_layout()
            plt.savefig(f'{config.RESULTS_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature importance analysis completed")
            return importance_df
            
        except Exception as e:
            logger.error(f"Error in feature importance analysis: {e}")
            return pd.DataFrame()
    
    def cross_validation_analysis(self) -> Dict[str, Dict[str, float]]:
        """Perform comprehensive cross-validation analysis"""
        try:
            X, y, _ = self.prepare_features()
            
            if X.empty or y.empty:
                return {}
            
            cv_results = {}
            
            for name, model in self.models.items():
                if name == 'neural_network':
                    continue  # Skip NN for CV (requires special handling)
                
                logger.info(f"Cross-validating {name}...")
                
                # Determine if scaling is needed
                if name in ['svr', 'ridge']:
                    X_cv = self.scalers['robust'].fit_transform(X)
                else:
                    X_cv = X
                
                scores = cross_validate(
                    model, X_cv, y,
                    cv=config.CV_FOLDS,
                    scoring=['neg_mean_squared_error', 'r2'],
                    return_train_score=True
                )
                
                cv_results[name] = {
                    'test_rmse_mean': np.sqrt(-scores['test_neg_mean_squared_error']).mean(),
                    'test_rmse_std': np.sqrt(-scores['test_neg_mean_squared_error']).std(),
                    'test_r2_mean': scores['test_r2'].mean(),
                    'test_r2_std': scores['test_r2'].std(),
                    'train_rmse_mean': np.sqrt(-scores['train_neg_mean_squared_error']).mean(),
                    'train_r2_mean': scores['train_r2'].mean(),
                    'overfitting_score': scores['train_r2'].mean() - scores['test_r2'].mean()
                }
            
            logger.info("Cross-validation analysis completed")
            return cv_results
            
        except Exception as e:
            logger.error(f"Error in cross-validation analysis: {e}")
            return {}
    
    def _plot_model_comparison(self, results: Dict[str, Dict[str, float]]):
        """Plot model performance comparison"""
        try:
            metrics = ['r2', 'rmse', 'mae']
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for i, metric in enumerate(metrics):
                model_names = list(results.keys())
                values = [results[name][metric] for name in model_names]
                
                axes[i].bar(model_names, values, color='skyblue', alpha=0.7)
                axes[i].set_title(f'{metric.upper()} Comparison')
                axes[i].set_ylabel(metric.upper())
                axes[i].tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for j, v in enumerate(values):
                    axes[i].text(j, v, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(f'{config.RESULTS_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting model comparison: {e}")
    
    def _plot_predictions_vs_actual(self, predictions: Dict[str, np.ndarray], y_test: pd.Series):
        """Plot predictions vs actual values"""
        try:
            n_models = len(predictions)
            cols = min(3, n_models)
            rows = (n_models + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if n_models == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes.flatten()
            else:
                axes = axes.flatten()
            
            for i, (name, y_pred) in enumerate(predictions.items()):
                ax = axes[i] if i < len(axes) else axes[-1]
                
                # Scatter plot
                ax.scatter(y_test, y_pred, alpha=0.6, s=30)
                
                # Perfect prediction line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
                
                # Labels and title
                ax.set_xlabel('Actual Stress Score')
                ax.set_ylabel('Predicted Stress Score')
                ax.set_title(f'{name} - R² = {r2_score(y_test, y_pred):.3f}')
                
                # Add grid
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_models, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(f'{config.RESULTS_DIR}/predictions_vs_actual.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting predictions vs actual: {e}")


class EnsemblePredictor:
    """Advanced ensemble methods for improved predictions"""
    
    def __init__(self):
        self.base_models = {}
        self.meta_model = None
        self.is_trained = False
        logger.info("EnsemblePredictor initialized")
    
    def create_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series) -> np.ndarray:
        """Create stacked ensemble for improved predictions"""
        try:
            # Base models
            base_models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
                'xgb': xgb.XGBRegressor(n_estimators=100, random_state=config.RANDOM_STATE, verbosity=0),
                'ridge': Ridge(alpha=1.0)
            }
            
            # Scale data for Ridge regression
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            self.scaler = scaler
            
            # Create base predictions using cross-validation
            base_predictions = np.zeros((X_train.shape[0], len(base_models)))
            
            for i, (name, model) in enumerate(base_models.items()):
                logger.info(f"Training base model: {name}")
                
                # Use scaled data for Ridge, original for tree-based models
                X_cv = X_train_scaled if name == 'ridge' else X_train
                
                # Cross-validation predictions
                cv_preds = cross_val_predict(model, X_cv, y_train, cv=5)
                base_predictions[:, i] = cv_preds
                
                # Train on full dataset
                model.fit(X_cv, y_train)
                self.base_models[name] = model
            
            # Meta-model (learns to combine base model predictions)
            self.meta_model = Ridge(alpha=0.1)
            self.meta_model.fit(base_predictions, y_train)
            
            self.is_trained = True
            logger.info("Stacking ensemble trained successfully")
            
            return base_predictions
            
        except Exception as e:
            logger.error(f"Error creating stacking ensemble: {e}")
            return np.array([])
    
    def predict_ensemble(self, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions"""
        try:
            if not self.is_trained:
                logger.error("Ensemble model not trained")
                return np.array([]), np.array([])
            
            base_preds = np.zeros((X_test.shape[0], len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                # Use scaled data for Ridge, original for tree-based models
                X_pred = self.scaler.transform(X_test) if name == 'ridge' else X_test
                base_preds[:, i] = model.predict(X_pred)
            
            # Meta-model prediction
            final_pred = self.meta_model.predict(base_preds)
            
            return final_pred, base_preds
            
        except Exception as e:
            logger.error(f"Error making ensemble predictions: {e}")
            return np.array([]), np.array([])


class RealTimeStressMonitor:
    """Real-time audio processing and stress prediction"""
    
    def __init__(self, model, scaler, feature_extractor: SoundscapeAnalyzer):
        self.model = model
        self.scaler = scaler
        self.feature_extractor = feature_extractor
        self.audio_buffer = []
        self.buffer_duration = config.AUDIO_DURATION * config.SAMPLE_RATE
        logger.info("RealTimeStressMonitor initialized")
    
    def process_audio_stream(self, audio_chunk: np.ndarray) -> Tuple[Optional[float], Optional[Dict[str, float]]]:
        """Process real-time audio stream for stress prediction"""
        try:
            # Add to buffer
            self.audio_buffer.extend(audio_chunk.tolist())
            
            # Process when buffer is full
            if len(self.audio_buffer) >= self.buffer_duration:
                # Extract features from current buffer
                audio_segment = np.array(self.audio_buffer[:self.buffer_duration])
                features = self.feature_extractor.extract_acoustic_features(audio_segment)
                
                if not features:
                    logger.warning("No features extracted from audio")
                    return None, None
                
                # Add temporal context
                current_time = datetime.now()
                current_hour = current_time.hour
                is_rush_hour = 1 if (7 <= current_hour <= 9) or (17 <= current_hour <= 19) else 0
                is_weekend = 1 if current_time.weekday() >= 5 else 0
                
                # Create feature vector (must match training features)
                feature_vector = self._create_feature_vector(features, current_hour, is_rush_hour, is_weekend)
                
                if feature_vector is not None:
                    # Scale features
                    feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
                    
                    # Predict stress level
                    stress_prediction = self.model.predict(feature_vector_scaled)[0]
                    
                    # Clear processed part of buffer (with overlap)
                    overlap_samples = self.buffer_duration // 2
                    self.audio_buffer = self.audio_buffer[overlap_samples:]
                    
                    return stress_prediction, features
                
            return None, None
            
        except Exception as e:
            logger.error(f"Error processing audio stream: {e}")
            return None, None
    
    def _create_feature_vector(self, features: Dict[str, float], hour: int, is_rush_hour: int, is_weekend: int) -> Optional[np.ndarray]:
        """Create feature vector from acoustic features and context"""
        try:
            # Essential features (must be present)
            essential_features = [
                'spl_mean', 'spectral_centroid_mean', 'roughness', 'sharpness',
                'zcr_mean', 'rms_energy'
            ]
            
            feature_values = []
            for feature in essential_features:
                if feature in features:
                    feature_values.append(features[feature])
                else:
                    logger.warning(f"Missing feature: {feature}")
                    return None
            
            # Add temporal features
            feature_values.extend([hour, is_rush_hour, is_weekend])
            
            # Add MFCC features (first 5 coefficients)
            for i in range(5):
                mfcc_mean = features.get(f'mfcc_{i}_mean', 0.0)
                feature_values.append(mfcc_mean)
            
            return np.array(feature_values)
            
        except Exception as e:
            logger.error(f"Error creating feature vector: {e}")
            return None


class PolicyImpactSimulator:
    """Simulate impact of noise reduction policies"""
    
    def __init__(self, trained_model, df: pd.DataFrame):
        self.model = trained_model
        self.df = df
        logger.info("PolicyImpactSimulator initialized")
    
    def simulate_policies(self) -> Dict[str, Dict[str, float]]:
        """Simulate impact of various noise reduction policies"""
        try:
            policies = {
                'traffic_reduction_10%': {'spl_reduction': 2.0, 'cost_per_person': 50},
                'traffic_reduction_25%': {'spl_reduction': 5.0, 'cost_per_person': 150},
                'green_sound_barriers': {'spl_reduction': 3.0, 'cost_per_person': 30},
                'quiet_hours_enforcement': {'spl_reduction': 4.0, 'cost_per_person': 20},
                'electric_vehicle_incentives': {'spl_reduction': 1.5, 'cost_per_person': 100},
                'improved_road_surfaces': {'spl_reduction': 2.5, 'cost_per_person': 75}
            }
            
            results = {}
            
            # Original stress level
            original_stress = self.df['composite_stress_score'].mean()
            
            for policy_name, policy_params in policies.items():
                logger.info(f"Simulating policy: {policy_name}")
                
                # Simulate reduced noise levels
                simulated_df = self.df.copy()
                simulated_df['spl_mean'] = np.maximum(
                    simulated_df['spl_mean'] - policy_params['spl_reduction'],
                    30.0  # Minimum realistic noise level
                )
                
                # Predict new stress levels (simplified prediction)
                stress_reduction_rate = policy_params['spl_reduction'] / 10.0  # Approximate relationship
                new_stress = original_stress * (1 - stress_reduction_rate * 0.1)  # 10% reduction per 10dB
                stress_reduction = original_stress - new_stress
                
                # Calculate impact metrics
                population = len(self.df)
                total_cost = population * policy_params['cost_per_person']
                cost_per_stress_point = total_cost / (stress_reduction * population) if stress_reduction > 0 else float('inf')
                
                results[policy_name] = {
                    'noise_reduction_db': policy_params['spl_reduction'],
                    'stress_reduction': stress_reduction,
                    'stress_reduction_percentage': (stress_reduction / original_stress) * 100,
                    'total_cost': total_cost,
                    'cost_per_person': policy_params['cost_per_person'],
                    'cost_per_stress_point': cost_per_stress_point,
                    'affected_population': population,
                    'roi_score': stress_reduction / policy_params['cost_per_person'] if policy_params['cost_per_person'] > 0 else 0,
                    'cost_effectiveness_rank': 0  # Will be filled later
                }
            
            # Rank policies by cost-effectiveness
            sorted_policies = sorted(results.items(), key=lambda x: x[1]['roi_score'], reverse=True)
            for rank, (policy_name, _) in enumerate(sorted_policies, 1):
                results[policy_name]['cost_effectiveness_rank'] = rank
            
            # Create visualization
            self._visualize_policy_impact(results)
            
            logger.info("Policy simulation completed")
            return results
            
        except Exception as e:
            logger.error(f"Error simulating policies: {e}")
            return {}
    
    def generate_recommendations(self, policy_results: Dict[str, Dict[str, float]]) -> List[Dict[str, str]]:
        """Generate data-driven policy recommendations"""
        try:
            recommendations = []
            
            # Sort by ROI score
            sorted_policies = sorted(policy_results.items(), 
                                   key=lambda x: x[1]['roi_score'], reverse=True)
            
            for policy_name, results in sorted_policies:
                if results['stress_reduction'] > 0:
                    priority = 'High' if results['roi_score'] > 0.02 else 'Medium' if results['roi_score'] > 0.01 else 'Low'
                    
                    recommendations.append({
                        'policy': policy_name.replace('_', ' ').title(),
                        'priority': priority,
                        'expected_impact': f"{results['stress_reduction_percentage']:.1f}% stress reduction",
                        'cost_effectiveness': f"${results['cost_per_stress_point']:.0f} per stress point reduced",
                        'noise_reduction': f"{results['noise_reduction_db']:.1f} dB reduction",
                        'cost_per_person': f"${results['cost_per_person']:.0f} per person",
                        'roi_score': f"{results['roi_score']:.4f}",
                        'rank': str(results['cost_effectiveness_rank'])
                    })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _visualize_policy_impact(self, results: Dict[str, Dict[str, float]]):
        """Visualize policy impact analysis"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            policies = list(results.keys())
            policies_clean = [p.replace('_', ' ').title() for p in policies]
            
            # Stress reduction comparison
            stress_reductions = [results[p]['stress_reduction_percentage'] for p in policies]
            ax1.bar(policies_clean, stress_reductions, color='lightcoral', alpha=0.7)
            ax1.set_title('Expected Stress Reduction by Policy')
            ax1.set_ylabel('Stress Reduction (%)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Cost comparison
            costs = [results[p]['cost_per_person'] for p in policies]
            ax2.bar(policies_clean, costs, color='lightblue', alpha=0.7)
            ax2.set_title('Implementation Cost per Person')
            ax2.set_ylabel('Cost ($)')
            ax2.tick_params(axis='x', rotation=45)
            
            # ROI comparison
            roi_scores = [results[p]['roi_score'] for p in policies]
            ax3.bar(policies_clean, roi_scores, color='lightgreen', alpha=0.7)
            ax3.set_title('Return on Investment Score')
            ax3.set_ylabel('ROI Score')
            ax3.tick_params(axis='x', rotation=45)
            
            # Cost-effectiveness scatter
            ax4.scatter(costs, stress_reductions, s=100, alpha=0.7, color='purple')
            for i, policy in enumerate(policies_clean):
                ax4.annotate(policy, (costs[i], stress_reductions[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.set_xlabel('Cost per Person ($)')
            ax4.set_ylabel('Stress Reduction (%)')
            ax4.set_title('Cost-Effectiveness Analysis')
            
            plt.tight_layout()
            plt.savefig(f'{config.RESULTS_DIR}/policy_impact_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error visualizing policy impact: {e}")


class ComprehensiveAnalyticsPipeline:
    """Main pipeline orchestrating the entire analysis"""
    
    def __init__(self):
        self.soundscape_analyzer = SoundscapeAnalyzer()
        self.health_processor = MentalHealthDataProcessor()
        self.env_integrator = EnvironmentalDataIntegrator()
        self.results = {}
        logger.info("ComprehensiveAnalyticsPipeline initialized")
    
    def run_complete_analysis(self, data_file: str) -> Dict[str, Any]:
        """Run the complete noise-mental health analysis pipeline"""
        try:
            logger.info("Starting comprehensive analysis pipeline...")
            
            # 1. Load and preprocess data
            logger.info("Step 1: Loading and preprocessing data")
            df = self._load_and_preprocess_data(data_file)
            
            if df.empty:
                logger.error("No data available for analysis")
                return {}
            
            # 2. Exploratory Data Analysis
            logger.info("Step 2: Exploratory Data Analysis")
            analyzer = NoiseHealthAnalyzer(df)
            
            correlation_results = analyzer.correlation_analysis()
            temporal_results = analyzer.temporal_pattern_analysis()
            statistical_results = analyzer.statistical_analysis()
            
            # 3. Machine Learning Models
            logger.info("Step 3: Training Machine Learning Models")
            ml_pipeline = StressPredictionModels(df)
            model_results = ml_pipeline.train_models()
            feature_importance = ml_pipeline.feature_importance_analysis()
            cv_results = ml_pipeline.cross_validation_analysis()
            
            # 4. Ensemble Methods
            logger.info("Step 4: Training Ensemble Models")
            ensemble = EnsemblePredictor()
            X, y, _ = ml_pipeline.prepare_features()
            
            if not X.empty and not y.empty:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
                )
                base_preds = ensemble.create_stacking_ensemble(X_train, y_train)
                ensemble_preds, _ = ensemble.predict_ensemble(X_test)
                
                if len(ensemble_preds) > 0:
                    ensemble_r2 = r2_score(y_test, ensemble_preds)
                    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_preds))
                    logger.info(f"Ensemble Model - R²: {ensemble_r2:.4f}, RMSE: {ensemble_rmse:.4f}")
            
            # 5. Policy Impact Analysis
            logger.info("Step 5: Policy Impact Analysis")
            if 'random_forest' in ml_pipeline.models:
                policy_simulator = PolicyImpactSimulator(ml_pipeline.models['random_forest'], df)
                policy_results = policy_simulator.simulate_policies()
                policy_recommendations = policy_simulator.generate_recommendations(policy_results)
            else:
                policy_results = {}
                policy_recommendations = []
            
            # 6. Compile Results
            self.results = {
                'data_summary': {
                    'total_records': len(df),
                    'features_count': len(df.columns),
                    'date_range': f"{df['timestamp'].min()} to {df['timestamp'].max()}" if 'timestamp' in df.columns else 'N/A'
                },
                'correlation_analysis': correlation_results.to_dict() if not correlation_results.empty else {},
                'temporal_patterns': temporal_results,
                'statistical_tests': statistical_results,
                'model_performance': model_results,
                'feature_importance': feature_importance.to_dict() if not feature_importance.empty else {},
                'cross_validation': cv_results,
                'policy_analysis': policy_results,
                'policy_recommendations': policy_recommendations
            }
            
            # 7. Generate Report
            logger.info("Step 6: Generating Analysis Report")
            self._generate_analysis_report()
            
            logger.info("Comprehensive analysis completed successfully!")
            return self.results
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return {}
    
    def _load_and_preprocess_data(self, data_file: str) -> pd.DataFrame:
        """Load and preprocess the input data"""
        try:
            # Try to load data file
            if os.path.exists(data_file):
                if data_file.endswith('.csv'):
                    df = pd.read_csv(data_file)
                elif data_file.endswith('.json'):
                    df = pd.read_json(data_file)
                else:
                    logger.error(f"Unsupported file format: {data_file}")
                    return pd.DataFrame()
            else:
                # Generate synthetic data for demonstration
                logger.info("Data file not found. Generating synthetic data for demonstration.")
                df = self._generate_synthetic_data()
            
            # Process health data
            df = self.health_processor.process_survey_data(df)
            
            # Add temporal features
            df = self.env_integrator.add_temporal_features(df)
            
            # Add urban context
            df = self.env_integrator.add_urban_context(df)
            
            logger.info(f"Data preprocessing completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading and preprocessing data: {e}")
            return pd.DataFrame()
    
    def _generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate synthetic data for demonstration purposes"""
        try:
            logger.info(f"Generating {n_samples} synthetic data points...")
            
            np.random.seed(config.RANDOM_STATE)
            
            # Generate timestamps
            start_date = datetime.now() - timedelta(days=30)
            timestamps = [start_date + timedelta(hours=x) for x in range(n_samples)]
            
            # Generate acoustic features with realistic correlations
            base_noise = np.random.normal(65, 10, n_samples)  # Base noise level ~65dB
            spl_mean = np.clip(base_noise, 35, 95)
            
            # Correlated acoustic features
            spectral_centroid_mean = np.random.normal(2000 + spl_mean * 10, 500)
            roughness = np.random.exponential(0.1) * (spl_mean / 65)
            sharpness = np.random.gamma(2, 0.5) * (spl_mean / 65)
            zcr_mean = np.random.normal(0.1, 0.02)
            rms_energy = np.random.normal(0.5, 0.1) * (spl_mean / 65)
            
            # Generate MFCC features
            mfcc_features = {}
            for i in range(config.N_MFCC):
                mfcc_features[f'mfcc_{i}_mean'] = np.random.normal(0, 1, n_samples)
                mfcc_features[f'mfcc_{i}_std'] = np.random.exponential(0.5, n_samples)
            
            # Generate health indicators (correlated with noise)
            noise_stress_factor = (spl_mean - 50) / 20  # Normalize noise impact
            sleep_quality = np.clip(8 - noise_stress_factor * 2 + np.random.normal(0, 1, n_samples), 1, 10)
            anxiety_level = np.clip(3 + noise_stress_factor * 1.5 + np.random.normal(0, 1, n_samples), 1, 10)
            mood_rating = np.clip(7 - noise_stress_factor * 1.2 + np.random.normal(0, 1, n_samples), 1, 10)
            depression_score = np.clip(2 + noise_stress_factor * 0.8 + np.random.normal(0, 0.5, n_samples), 1, 10)
            
            # Demographics
            ages = np.random.randint(18, 80, n_samples)
            occupations = np.random.choice(['healthcare', 'education', 'finance', 'technology', 'retail', 'other'], n_samples)
            
            # Environmental factors
            temperatures = np.random.normal(20, 10, n_samples)
            humidity = np.random.normal(60, 15, n_samples)
            wind_speed = np.random.exponential(5, n_samples)
            precipitation = np.random.exponential(2, n_samples)
            
            # Urban context
            population = np.random.randint(1000, 100000, n_samples)
            area_km2 = np.random.uniform(1, 50, n_samples)
            daily_traffic_count = np.random.randint(500, 50000, n_samples)
            
            # Create DataFrame
            data = {
                'timestamp': timestamps,
                'spl_mean': spl_mean,
                'spl_std': np.random.normal(5, 2, n_samples),
                'spectral_centroid_mean': spectral_centroid_mean,
                'spectral_centroid_std': np.random.normal(200, 50, n_samples),
                'roughness': roughness,
                'sharpness': sharpness,
                'zcr_mean': zcr_mean,
                'zcr_std': np.random.normal(0.02, 0.005, n_samples),
                'rms_energy': rms_energy,
                'chroma_mean': np.random.normal(0.5, 0.1, n_samples),
                'chroma_std': np.random.normal(0.2, 0.05, n_samples),
                'loudness': np.random.normal(0.3, 0.1, n_samples) * (spl_mean / 65),
                'sleep_quality': sleep_quality,
                'anxiety_level': anxiety_level,
                'mood_rating': mood_rating,
                'depression_score': depression_score,
                'age': ages,
                'occupation': occupations,
                'temperature': temperatures,
                'humidity': humidity,
                'wind_speed': wind_speed,
                'precipitation': precipitation,
                'population': population,
                'area_km2': area_km2,
                'daily_traffic_count': daily_traffic_count
            }
            
            # Add MFCC features to data
            data.update(mfcc_features)
            
            df = pd.DataFrame(data)
            
            logger.info("Synthetic data generated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return pd.DataFrame()
    
    def _generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        try:
            report_content = []
            
            # Header
            report_content.append("# Urban Noise Pollution Impact on Mental Health - Analysis Report")
            report_content.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_content.append("")
            
            # Executive Summary
            report_content.append("## Executive Summary")
            data_summary = self.results.get('data_summary', {})
            report_content.append(f"- **Total Records Analyzed**: {data_summary.get('total_records', 'N/A')}")
            report_content.append(f"- **Features Used**: {data_summary.get('features_count', 'N/A')}")
            report_content.append(f"- **Data Period**: {data_summary.get('date_range', 'N/A')}")
            report_content.append("")
            
            # Key Findings
            report_content.append("## Key Findings")
            
            # Statistical significance
            stats_results = self.results.get('statistical_tests', {})
            corr_test = stats_results.get('correlation_test', {})
            if corr_test:
                correlation = corr_test.get('correlation', 0)
                significant = corr_test.get('significant', False)
                report_content.append(f"- **Noise-Stress Correlation**: {correlation:.3f} ({'Significant' if significant else 'Not Significant'})")
            
            # Best performing model
            model_results = self.results.get('model_performance', {})
            if model_results:
                best_model = max(model_results.keys(), key=lambda k: model_results[k].get('r2', 0))
                best_r2 = model_results[best_model].get('r2', 0)
                report_content.append(f"- **Best Prediction Model**: {best_model.title()} (R² = {best_r2:.3f})")
            
            # Group differences
            group_comp = stats_results.get('group_comparison', {})
            if group_comp:
                effect_size = group_comp.get('effect_size', 0)
                report_content.append(f"- **High vs Low Noise Effect Size**: {effect_size:.3f}")
            
            report_content.append("")
            
            # Model Performance
            report_content.append("## Model Performance Summary")
            report_content.append("| Model | R² Score | RMSE | MAE |")
            report_content.append("|-------|----------|------|-----|")
            
            for model_name, metrics in model_results.items():
                r2 = metrics.get('r2', 0)
                rmse = metrics.get('rmse', 0)
                mae = metrics.get('mae', 0)
                report_content.append(f"| {model_name.title()} | {r2:.3f} | {rmse:.3f} | {mae:.3f} |")
            
            report_content.append("")
            
            # Feature Importance
            report_content.append("## Top Important Features")
            feature_importance = self.results.get('feature_importance', {})
            if feature_importance and 'feature' in feature_importance:
                features = list(feature_importance['feature'].values())[:10]
                importances = list(feature_importance['importance'].values())[:10]
                
                for feature, importance in zip(features, importances):
                    report_content.append(f"- **{feature}**: {importance:.4f}")
            
            report_content.append("")
            
            # Policy Recommendations
            report_content.append("## Policy Recommendations")
            recommendations = self.results.get('policy_recommendations', [])
            
            for i, rec in enumerate(recommendations[:5], 1):  # Top 5 recommendations
                report_content.append(f"### {i}. {rec.get('policy', 'N/A')} ({rec.get('priority', 'N/A')} Priority)")
                report_content.append(f"- **Expected Impact**: {rec.get('expected_impact', 'N/A')}")
                report_content.append(f"- **Cost Effectiveness**: {rec.get('cost_effectiveness', 'N/A')}")
                report_content.append(f"- **Noise Reduction**: {rec.get('noise_reduction', 'N/A')}")
                report_content.append(f"- **Cost per Person**: {rec.get('cost_per_person', 'N/A')}")
                report_content.append("")
            
            # Technical Details
            report_content.append("## Technical Details")
            report_content.append("### Cross-Validation Results")
            cv_results = self.results.get('cross_validation', {})
            
            if cv_results:
                report_content.append("| Model | CV R² Mean | CV RMSE Mean | Overfitting Score |")
                report_content.append("|-------|------------|--------------|-------------------|")
                
                for model_name, cv_metrics in cv_results.items():
                    r2_mean = cv_metrics.get('test_r2_mean', 0)
                    rmse_mean = cv_metrics.get('test_rmse_mean', 0)
                    overfitting = cv_metrics.get('overfitting_score', 0)
                    report_content.append(f"| {model_name.title()} | {r2_mean:.3f} | {rmse_mean:.3f} | {overfitting:.3f} |")
            
            report_content.append("")
            
            # Methodology
            report_content.append("## Methodology")
            report_content.append("### Data Processing")
            report_content.append("- Acoustic feature extraction using librosa and custom psychoacoustic algorithms")
            report_content.append("- Mental health survey data normalization and composite scoring")
            report_content.append("- Temporal and environmental context integration")
            report_content.append("")
            
            report_content.append("### Machine Learning Models")
            report_content.append("- Random Forest Regressor")
            report_content.append("- Gradient Boosting Regressor")
            report_content.append("- XGBoost Regressor")
            report_content.append("- Ridge Regression")
            report_content.append("- Support Vector Regression")
            if TF_AVAILABLE:
                report_content.append("- Neural Network (Multi-layer Perceptron)")
            report_content.append("")
            
            report_content.append("### Evaluation Metrics")
            report_content.append("- R² Score (Coefficient of Determination)")
            report_content.append("- Root Mean Square Error (RMSE)")
            report_content.append("- Mean Absolute Error (MAE)")
            report_content.append("- 5-fold Cross-Validation")
            report_content.append("")
            
            # Save report
            report_text = "\n".join(report_content)
            
            with open(f'{config.RESULTS_DIR}/analysis_report.md', 'w') as f:
                f.write(report_text)
            
            # Also save as JSON for programmatic access
            with open(f'{config.RESULTS_DIR}/analysis_results.json', 'w') as f:
                # Convert numpy types to native Python types for JSON serialization
                json_results = self._convert_numpy_types(self.results)
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info("Analysis report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating analysis report: {e}")
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to native Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj


class InteractiveDashboard:
    """Create interactive dashboard for noise-mental health analytics"""
    
    def __init__(self, df: pd.DataFrame, results: Dict[str, Any]):
        self.df = df
        self.results = results
        logger.info("InteractiveDashboard initialized")
    
    def create_plotly_dashboard(self) -> Dict[str, go.Figure]:
        """Create interactive Plotly visualizations"""
        try:
            figures = {}
            
            # 1. Time Series Plot
            if 'timestamp' in self.df.columns:
                fig_timeseries = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Noise Levels Over Time', 'Stress Levels Over Time'),
                    vertical_spacing=0.1
                )
                
                # Noise levels
                fig_timeseries.add_trace(
                    go.Scatter(
                        x=self.df['timestamp'],
                        y=self.df['spl_mean'],
                        mode='lines',
                        name='Noise Level (dB)',
                        line=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Stress levels
                fig_timeseries.add_trace(
                    go.Scatter(
                        x=self.df['timestamp'],
                        y=self.df['composite_stress_score'],
                        mode='lines',
                        name='Stress Score',
                        line=dict(color='red')
                    ),
                    row=2, col=1
                )
                
                fig_timeseries.update_layout(
                    title='Noise and Stress Trends Over Time',
                    height=600,
                    showlegend=True
                )
                
                figures['timeseries'] = fig_timeseries
            
            # 2. Correlation Heatmap
            acoustic_cols = [col for col in self.df.columns if any(x in col for x in 
                           ['spl', 'spectral', 'roughness', 'sharpness', 'loudness'])][:10]  # Top 10
            health_cols = ['composite_stress_score', 'anxiety_level', 'sleep_quality', 'mood_rating']
            health_cols = [col for col in health_cols if col in self.df.columns]
            
            if acoustic_cols and health_cols:
                corr_data = self.df[acoustic_cols + health_cols].corr()
                
                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr_data.values,
                    x=corr_data.columns,
                    y=corr_data.columns,
                    colorscale='RdBu',
                    zmid=0,
                    text=np.round(corr_data.values, 3),
                    texttemplate='%{text}',
                    textfont={"size": 10},
                    hoverongaps=False
                ))
                
                fig_heatmap.update_layout(
                    title='Feature Correlation Matrix',
                    width=800,
                    height=600
                )
                
                figures['correlation'] = fig_heatmap
            
            # 3. Scatter Plot: Noise vs Stress
            fig_scatter = go.Figure()
            
            # Add color coding by time of day if available
            if 'hour' in self.df.columns:
                fig_scatter.add_trace(go.Scatter(
                    x=self.df['spl_mean'],
                    y=self.df['composite_stress_score'],
                    mode='markers',
                    marker=dict(
                        color=self.df['hour'],
                        colorscale='Viridis',
                        size=8,
                        colorbar=dict(title="Hour of Day"),
                        opacity=0.7
                    ),
                    text=self.df['hour'],
                    hovertemplate='<b>Noise</b>: %{x:.1f} dB<br>' +
                                 '<b>Stress</b>: %{y:.2f}<br>' +
                                 '<b>Hour</b>: %{text}<br>' +
                                 '<extra></extra>',
                    name='Data Points'
                ))
            else:
                fig_scatter.add_trace(go.Scatter(
                    x=self.df['spl_mean'],
                    y=self.df['composite_stress_score'],
                    mode='markers',
                    marker=dict(size=8, opacity=0.7),
                    name='Data Points'
                ))
            
            # Add trend line
            z = np.polyfit(self.df['spl_mean'], self.df['composite_stress_score'], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(self.df['spl_mean'].min(), self.df['spl_mean'].max(), 100)
            
            fig_scatter.add_trace(go.Scatter(
                x=x_trend,
                y=p(x_trend),
                mode='lines',
                name='Trend Line',
                line=dict(color='red', width=2)
            ))
            
            fig_scatter.update_layout(
                title='Noise Level vs Stress Score Relationship',
                xaxis_title='Noise Level (dB)',
                yaxis_title='Stress Score',
                width=800,
                height=600
            )
            
            figures['scatter'] = fig_scatter
            
            # 4. Box Plot: Stress by Categories
            if 'age_group' in self.df.columns:
                fig_box = go.Figure()
                
                for age_group in self.df['age_group'].unique():
                    if pd.notna(age_group):
                        group_data = self.df[self.df['age_group'] == age_group]['composite_stress_score']
                        fig_box.add_trace(go.Box(
                            y=group_data,
                            name=str(age_group),
                            boxmean='sd'
                        ))
                
                fig_box.update_layout(
                    title='Stress Distribution by Age Group',
                    xaxis_title='Age Group',
                    yaxis_title='Stress Score',
                    width=800,
                    height=500
                )
                
                figures['boxplot'] = fig_box
            
            # 5. Model Performance Comparison
            model_results = self.results.get('model_performance', {})
            if model_results:
                models = list(model_results.keys())
                r2_scores = [model_results[model].get('r2', 0) for model in models]
                rmse_scores = [model_results[model].get('rmse', 0) for model in models]
                
                fig_performance = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('R² Scores', 'RMSE Scores'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig_performance.add_trace(
                    go.Bar(x=models, y=r2_scores, name='R² Score', marker_color='lightblue'),
                    row=1, col=1
                )
                
                fig_performance.add_trace(
                    go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='lightcoral'),
                    row=1, col=2
                )
                
                fig_performance.update_layout(
                    title='Model Performance Comparison',
                    height=500,
                    showlegend=False
                )
                
                figures['performance'] = fig_performance
            
            # 6. Policy Impact Visualization
            policy_results = self.results.get('policy_analysis', {})
            if policy_results:
                policies = list(policy_results.keys())
                policies_clean = [p.replace('_', ' ').title() for p in policies]
                stress_reductions = [policy_results[p]['stress_reduction_percentage'] for p in policies]
                costs = [policy_results[p]['cost_per_person'] for p in policies]
                roi_scores = [policy_results[p]['roi_score'] for p in policies]
                
                fig_policy = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Stress Reduction by Policy', 'Cost per Person', 
                                   'ROI Scores', 'Cost vs Effectiveness'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Stress reduction
                fig_policy.add_trace(
                    go.Bar(x=policies_clean, y=stress_reductions, name='Stress Reduction %', marker_color='lightgreen'),
                    row=1, col=1
                )
                
                # Costs
                fig_policy.add_trace(
                    go.Bar(x=policies_clean, y=costs, name='Cost per Person', marker_color='lightblue'),
                    row=1, col=2
                )
                
                # ROI
                fig_policy.add_trace(
                    go.Bar(x=policies_clean, y=roi_scores, name='ROI Score', marker_color='lightyellow'),
                    row=2, col=1
                )
                
                # Scatter: Cost vs Effectiveness
                fig_policy.add_trace(
                    go.Scatter(
                        x=costs, y=stress_reductions,
                        mode='markers+text',
                        text=policies_clean,
                        textposition='top center',
                        marker=dict(size=10, color='purple'),
                        name='Policies'
                    ),
                    row=2, col=2
                )
                
                fig_policy.update_layout(
                    title='Policy Impact Analysis',
                    height=800,
                    showlegend=False
                )
                
                # Update x-axis labels
                for i in range(1, 3):
                    for j in range(1, 3):
                        if i < 2 or j < 2:  # Don't update scatter plot axes
                            fig_policy.update_xaxes(tickangle=45, row=i, col=j)
                
                figures['policy'] = fig_policy
            
            logger.info(f"Created {len(figures)} interactive visualizations")
            return figures
            
        except Exception as e:
            logger.error(f"Error creating Plotly dashboard: {e}")
            return {}
    
    def save_dashboard_html(self, figures: Dict[str, go.Figure]):
        """Save dashboard as HTML files"""
        try:
            for name, fig in figures.items():
                filename = f'{config.RESULTS_DIR}/dashboard_{name}.html'
                fig.write_html(filename)
                logger.info(f"Saved {name} dashboard to {filename}")
                
        except Exception as e:
            logger.error(f"Error saving dashboard HTML: {e}")


# ==================== UTILITY FUNCTIONS ====================

def setup_environment():
    """Setup the analysis environment"""
    try:
        # Create directories
        for directory in [config.DATA_DIR, config.MODELS_DIR, config.RESULTS_DIR, config.LOGS_DIR]:
            Path(directory).mkdir(exist_ok=True)
        
        logger.info("Environment setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up environment: {e}")
        return False


def save_model(model, model_name: str, metadata: Dict[str, Any] = None):
    """Save trained model with metadata"""
    try:
        import joblib
        
        model_path = f'{config.MODELS_DIR}/{model_name}.joblib'
        joblib.dump(model, model_path)
        
        # Save metadata
        if metadata:
            metadata_path = f'{config.MODELS_DIR}/{model_name}_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Model saved: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return False


def load_model(model_name: str):
    """Load trained model with metadata"""
    try:
        import joblib
        
        model_path = f'{config.MODELS_DIR}/{model_name}.joblib'
        model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = f'{config.MODELS_DIR}/{model_name}_metadata.json'
        metadata = None
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        
        logger.info(f"Model loaded: {model_path}")
        return model, metadata
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    try:
        logger.info("Starting Urban Noise Mental Health Analytics System")
        
        # Setup environment
        if not setup_environment():
            logger.error("Failed to setup environment")
            return
        
        # Initialize and run pipeline
        pipeline = ComprehensiveAnalyticsPipeline()
        
        # You can specify your data file here, or it will generate synthetic data
        data_file = "data/noise_mental_health_data.csv"  # Change this to your data file
        
        results = pipeline.run_complete_analysis(data_file)
        
        if results:
            logger.info("Analysis completed successfully!")
            
            # Create interactive dashboard
            dashboard = InteractiveDashboard(pipeline.health_processor.df if hasattr(pipeline, 'health_processor') else pd.DataFrame(), results)
            figures = dashboard.create_plotly_dashboard()
            dashboard.save_dashboard_html(figures)
            
            # Print summary
            print("\n" + "="*80)
            print("URBAN NOISE MENTAL HEALTH ANALYTICS - SUMMARY")
            print("="*80)
            
            data_summary = results.get('data_summary', {})
            print(f"📊 Records Analyzed: {data_summary.get('total_records', 'N/A')}")
            print(f"🔧 Features Used: {data_summary.get('features_count', 'N/A')}")
            
            # Best model
            model_results = results.get('model_performance', {})
            if model_results:
                best_model = max(model_results.keys(), key=lambda k: model_results[k].get('r2', 0))
                best_r2 = model_results[best_model].get('r2', 0)
                print(f"🏆 Best Model: {best_model.title()} (R² = {best_r2:.3f})")
            
            # Statistical significance
            stats_results = results.get('statistical_tests', {})
            corr_test = stats_results.get('correlation_test', {})
            if corr_test:
                correlation = corr_test.get('correlation', 0)
                significant = corr_test.get('significant', False)
                print(f"📈 Noise-Stress Correlation: {correlation:.3f} ({'✅ Significant' if significant else '❌ Not Significant'})")
            
            # Top recommendations
            recommendations = results.get('policy_recommendations', [])
            if recommendations:
                print(f"\n🎯 Top Policy Recommendations:")
                for i, rec in enumerate(recommendations[:3], 1):
                    print(f"   {i}. {rec.get('policy', 'N/A')} - {rec.get('expected_impact', 'N/A')}")
            
            print(f"\n📁 Results saved to: {config.RESULTS_DIR}/")
            print(f"📋 Full report: {config.RESULTS_DIR}/analysis_report.md")
            print("="*80)
            
        else:
            logger.error("Analysis failed")
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")


if __name__ == "__main__":
    main()


# ==================== EXAMPLE USAGE ====================

"""
EXAMPLE USAGE:

# Basic usage with synthetic data
from noise_mental_health_analytics import ComprehensiveAnalyticsPipeline

pipeline = ComprehensiveAnalyticsPipeline()
results = pipeline.run_complete_analysis("path/to/your/data.csv")

# Real-time monitoring
from noise_mental_health_analytics import RealTimeStressMonitor, SoundscapeAnalyzer
import joblib

# Load trained model
model = joblib.load('models/best_model.joblib')
scaler = joblib.load('models/scaler.joblib')
feature_extractor = SoundscapeAnalyzer()

monitor = RealTimeStressMonitor(model, scaler, feature_extractor)

# Process audio chunks (you would get these from microphone input)
import numpy as np
audio_chunk = np.random.randn(4410)  # 0.1 seconds at 44100 Hz
stress_level, features = monitor.process_audio_stream(audio_chunk)

if stress_level is not None:
    print(f"Predicted stress level: {stress_level:.2f}")

# Policy analysis
from noise_mental_health_analytics import PolicyImpactSimulator
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")
simulator = PolicyImpactSimulator(model, df)
policy_results = simulator.simulate_policies()
recommendations = simulator.generate_recommendations(policy_results)

for rec in recommendations:
    print(f"{rec['policy']}: {rec['expected_impact']} at {rec['cost_per_person']}")
"""