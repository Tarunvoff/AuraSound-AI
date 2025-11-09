#Urban Noise Pollution Impact on Mental Health Analytics
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

# Machine Learning - EXPANDED
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, cross_val_predict, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.linear_model import Ridge, LinearRegression, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE, RFECV
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.compose import ColumnTransformer

# Additional ML libraries for improved accuracy
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Audio Processing
import librosa
import soundfile as sf

# Deep Learning
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Dropout, Input, BatchNormalization, Attention
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1_l2
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Configuration
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Enhanced configuration for high-accuracy analysis"""
    
    # Audio Processing - IMPROVED
    SAMPLE_RATE: int = 44100
    AUDIO_DURATION: int = 10
    N_MFCC: int = 20  # Increased from 13
    N_CHROMA: int = 12
    N_MEL: int = 128
    
    # Machine Learning - ENHANCED
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 10  # Increased from 5 for better validation
    
    # Feature Engineering - EXPANDED
    SEQUENCE_LENGTH: int = 24
    USE_POLYNOMIAL_FEATURES: bool = True
    POLY_DEGREE: int = 2
    
    # High-accuracy thresholds
    HIGH_NOISE_THRESHOLD: float = 70.0
    HIGH_STRESS_THRESHOLD: float = 7.0
    
    # File paths
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    RESULTS_DIR: str = "results"
    LOGS_DIR: str = "logs"
    
    def __post_init__(self):
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            Path(dir_path).mkdir(exist_ok=True)

config = Config()

class EnhancedSoundscapeAnalyzer:
    """Significantly enhanced soundscape analysis for higher accuracy"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        self.sample_rate = sample_rate
        # Expanded frequency bands for better analysis
        self.frequency_bands = {
            'sub_bass': (20, 60),
            'bass': (60, 250),
            'low_mid': (250, 500),
            'mid': (500, 2000),
            'upper_mid': (2000, 4000),
            'presence': (4000, 8000),
            'brilliance': (8000, 20000)
        }
        logger.info(f"Enhanced SoundscapeAnalyzer initialized")
    
    def extract_comprehensive_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive acoustic features for maximum accuracy"""
        features = {}
        
        # Basic acoustic features (enhanced)
        features.update(self._extract_enhanced_basic_features(audio))
        
        # Advanced spectral features
        features.update(self._extract_advanced_spectral_features(audio))
        
        # Enhanced MFCC features
        features.update(self._extract_enhanced_mfcc_features(audio))
        
        # Psychoacoustic features (improved)
        features.update(self._extract_advanced_psychoacoustic_features(audio))
        
        # Temporal features (expanded)
        features.update(self._extract_expanded_temporal_features(audio))
        
        # NEW: Frequency band analysis
        features.update(self._extract_frequency_band_features(audio))
        
        # NEW: Statistical moments
        features.update(self._extract_statistical_moments(audio))
        
        # NEW: Harmonic analysis
        features.update(self._extract_harmonic_features(audio))
        
        return features
    
    def _extract_enhanced_basic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Enhanced basic acoustic features"""
        features = {}
        
        # Multiple SPL calculations
        rms = np.sqrt(np.mean(audio**2))
        features['spl_rms'] = 20 * np.log10(rms + 1e-10)
        features['spl_peak'] = 20 * np.log10(np.max(np.abs(audio)) + 1e-10)
        features['spl_percentile_90'] = 20 * np.log10(np.percentile(np.abs(audio), 90) + 1e-10)
        features['spl_percentile_50'] = 20 * np.log10(np.percentile(np.abs(audio), 50) + 1e-10)
        features['spl_percentile_10'] = 20 * np.log10(np.percentile(np.abs(audio), 10) + 1e-10)
        
        # Dynamic range
        features['dynamic_range'] = features['spl_peak'] - features['spl_percentile_10']
        
        # Crest factor
        features['crest_factor'] = np.max(np.abs(audio)) / rms if rms > 0 else 0
        
        # Energy in different segments
        n_segments = 10
        segment_length = len(audio) // n_segments
        segment_energies = []
        for i in range(n_segments):
            start = i * segment_length
            end = start + segment_length
            segment_energy = np.mean(audio[start:end]**2)
            segment_energies.append(segment_energy)
        
        features['energy_variance'] = np.var(segment_energies)
        features['energy_mean'] = np.mean(segment_energies)
        
        return features
    
    def _extract_advanced_spectral_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Advanced spectral analysis features"""
        features = {}
        
        # Standard spectral features with more statistics
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_skew'] = stats.skew(spectral_centroids)
        features['spectral_centroid_kurt'] = stats.kurtosis(spectral_centroids)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral rolloff (multiple percentiles)
        for p in [0.85, 0.90, 0.95]:
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate, roll_percent=p)[0]
            features[f'spectral_rolloff_{int(p*100)}_mean'] = np.mean(rolloff)
            features[f'spectral_rolloff_{int(p*100)}_std'] = np.std(rolloff)
        
        # Zero crossing rate with statistics
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        features['zcr_max'] = np.max(zcr)
        features['zcr_min'] = np.min(zcr)
        
        # Spectral flatness (measure of noise-like vs tonal content)
        spec_flat = librosa.feature.spectral_flatness(y=audio)[0]
        features['spectral_flatness_mean'] = np.mean(spec_flat)
        features['spectral_flatness_std'] = np.std(spec_flat)
        
        # Spectral contrast
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        for i in range(spec_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spec_contrast[i])
            features[f'spectral_contrast_{i}_std'] = np.std(spec_contrast[i])
        
        return features
    
    def _extract_enhanced_mfcc_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Enhanced MFCC feature extraction"""
        features = {}
        
        # Standard MFCCs with more coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=config.N_MFCC)
        
        for i in range(config.N_MFCC):
            mfcc_vals = mfccs[i]
            features[f'mfcc_{i}_mean'] = np.mean(mfcc_vals)
            features[f'mfcc_{i}_std'] = np.std(mfcc_vals)
            features[f'mfcc_{i}_max'] = np.max(mfcc_vals)
            features[f'mfcc_{i}_min'] = np.min(mfcc_vals)
            features[f'mfcc_{i}_skew'] = stats.skew(mfcc_vals)
            features[f'mfcc_{i}_kurt'] = stats.kurtosis(mfcc_vals)
        
        # Delta MFCCs (rate of change)
        mfcc_delta = librosa.feature.delta(mfccs)
        for i in range(min(5, config.N_MFCC)):  # First 5 delta MFCCs
            features[f'mfcc_delta_{i}_mean'] = np.mean(mfcc_delta[i])
            features[f'mfcc_delta_{i}_std'] = np.std(mfcc_delta[i])
        
        # Delta-delta MFCCs (acceleration)
        mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
        for i in range(min(3, config.N_MFCC)):  # First 3 delta-delta MFCCs
            features[f'mfcc_delta2_{i}_mean'] = np.mean(mfcc_delta2[i])
            features[f'mfcc_delta2_{i}_std'] = np.std(mfcc_delta2[i])
        
        # Enhanced chroma features
        chroma = librosa.feature.chroma(y=audio, sr=self.sample_rate, n_chroma=config.N_CHROMA)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.mean(np.std(chroma, axis=1))
        features['chroma_var'] = np.mean(np.var(chroma, axis=1))
        
        # Chroma energy normalized statistics
        chroma_norm = np.linalg.norm(chroma, axis=0)
        features['chroma_norm_mean'] = np.mean(chroma_norm)
        features['chroma_norm_std'] = np.std(chroma_norm)
        
        return features
    
    def _extract_frequency_band_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract features from specific frequency bands"""
        features = {}
        
        # Get the spectrogram
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=self.sample_rate)
        
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            # Find frequency bin indices
            low_bin = np.argmax(freqs >= low_freq)
            high_bin = np.argmax(freqs >= high_freq)
            
            if high_bin == 0:  # Handle edge case
                high_bin = len(freqs)
            
            # Extract magnitude for this frequency band
            band_magnitude = magnitude[low_bin:high_bin, :]
            
            if band_magnitude.size > 0:
                band_energy = np.sum(band_magnitude, axis=0)
                features[f'{band_name}_energy_mean'] = np.mean(band_energy)
                features[f'{band_name}_energy_std'] = np.std(band_energy)
                features[f'{band_name}_energy_max'] = np.max(band_energy)
                features[f'{band_name}_peak_freq'] = freqs[low_bin + np.argmax(np.mean(band_magnitude, axis=1))]
        
        return features
    
    def _extract_statistical_moments(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract statistical moments of the audio signal"""
        features = {}
        
        # Time domain moments
        features['audio_skewness'] = stats.skew(audio)
        features['audio_kurtosis'] = stats.kurtosis(audio)
        features['audio_moment_3'] = stats.moment(audio, moment=3)
        features['audio_moment_4'] = stats.moment(audio, moment=4)
        
        # Spectral domain moments
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        spectral_mean = np.mean(magnitude, axis=1)
        
        features['spectral_skewness'] = stats.skew(spectral_mean)
        features['spectral_kurtosis'] = stats.kurtosis(spectral_mean)
        
        return features
    
    def _extract_harmonic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Extract harmonic and percussive features"""
        features = {}
        
        try:
            # Harmonic-percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Harmonic features
            features['harmonic_energy'] = np.mean(y_harmonic**2)
            features['harmonic_zcr'] = np.mean(librosa.feature.zero_crossing_rate(y_harmonic)[0])
            
            # Percussive features
            features['percussive_energy'] = np.mean(y_percussive**2)
            features['percussive_zcr'] = np.mean(librosa.feature.zero_crossing_rate(y_percussive)[0])
            
            # Harmonic-to-percussive ratio
            h_energy = features['harmonic_energy']
            p_energy = features['percussive_energy']
            features['harmonic_percussive_ratio'] = h_energy / (p_energy + 1e-10)
            
        except Exception as e:
            logger.warning(f"Could not extract harmonic features: {e}")
            features['harmonic_energy'] = 0.0
            features['percussive_energy'] = 0.0
            features['harmonic_percussive_ratio'] = 1.0
        
        return features
    
    def _extract_advanced_psychoacoustic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Advanced psychoacoustic feature extraction"""
        features = {}
        
        # Enhanced roughness calculation
        features['roughness'] = self._calculate_advanced_roughness(audio)
        features['sharpness'] = self._calculate_advanced_sharpness(audio)
        features['loudness'] = self._calculate_advanced_loudness(audio)
        
        # Tonality (measure of tonal vs noise content)
        features['tonality'] = self._calculate_tonality(audio)
        
        return features
    
    def _extract_expanded_temporal_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Expanded temporal feature extraction"""
        features = {}
        
        # Enhanced onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sample_rate, units='time')
        features['onset_rate'] = len(onset_frames) / (len(audio) / self.sample_rate)
        
        if len(onset_frames) > 1:
            onset_intervals = np.diff(onset_frames)
            features['onset_interval_mean'] = np.mean(onset_intervals)
            features['onset_interval_std'] = np.std(onset_intervals)
        else:
            features['onset_interval_mean'] = 0.0
            features['onset_interval_std'] = 0.0
        
        # Enhanced tempo estimation
        try:
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = float(tempo)
            features['beat_count'] = len(beats)
            
            if len(beats) > 1:
                beat_intervals = np.diff(librosa.frames_to_time(beats, sr=self.sample_rate))
                features['beat_regularity'] = 1.0 / (1.0 + np.std(beat_intervals))
            else:
                features['beat_regularity'] = 0.0
                
        except:
            features['tempo'] = 0.0
            features['beat_count'] = 0
            features['beat_regularity'] = 0.0
        
        return features
    
    def _calculate_advanced_roughness(self, audio: np.ndarray) -> float:
        """Advanced roughness calculation"""
        try:
            # Multiple approaches to roughness
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Spectral irregularity
            spectral_diff = np.diff(magnitude, axis=0)
            roughness1 = np.mean(np.sum(spectral_diff**2, axis=0))
            
            # Modulation-based roughness
            envelope = np.abs(signal.hilbert(audio))
            mod_spectrum = np.abs(np.fft.fft(envelope))
            # Focus on modulation frequencies that contribute to roughness (15-300 Hz)
            mod_freqs = np.fft.fftfreq(len(envelope), 1/self.sample_rate)
            rough_indices = np.where((np.abs(mod_freqs) >= 15) & (np.abs(mod_freqs) <= 300))[0]
            roughness2 = np.mean(mod_spectrum[rough_indices]) if len(rough_indices) > 0 else 0.0
            
            return float((roughness1 + roughness2) / 2)
        except:
            return 0.0
    
    def _calculate_advanced_sharpness(self, audio: np.ndarray) -> float:
        """Advanced sharpness calculation"""
        try:
            freqs = librosa.fft_frequencies(sr=self.sample_rate)
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            
            # Weighted by frequency with perceptual weighting
            weights = (freqs / 1000.0) ** 1.5  # Perceptual weighting
            weighted_spectrum = magnitude * weights[:, np.newaxis]
            
            total_energy = np.sum(magnitude, axis=0)
            weighted_energy = np.sum(weighted_spectrum, axis=0)
            
            sharpness = np.mean(weighted_energy / (total_energy + 1e-10))
            return float(sharpness)
        except:
            return 0.0
    
    def _calculate_advanced_loudness(self, audio: np.ndarray) -> float:
        """Advanced loudness calculation using multiple models"""
        try:
            # A-weighted loudness
            freqs, times, spectrogram_data = signal.spectrogram(audio, self.sample_rate)
            a_weights = self._a_weighting(freqs)
            a_weighted_spec = spectrogram_data * a_weights[:, np.newaxis]
            a_loudness = np.mean(np.sum(a_weighted_spec, axis=0) ** 0.3)
            
            # RMS-based loudness
            rms_loudness = np.sqrt(np.mean(audio**2))
            
            # Combine both measures
            return float((a_loudness + rms_loudness) / 2)
        except:
            return 0.0
    
    def _calculate_tonality(self, audio: np.ndarray) -> float:
        """Calculate tonality measure"""
        try:
            # Use harmonic-percussive separation to measure tonality
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            harmonic_energy = np.mean(y_harmonic**2)
            total_energy = np.mean(audio**2)
            
            tonality = harmonic_energy / (total_energy + 1e-10)
            return float(tonality)
        except:
            return 0.5  # Default neutral value
    
    def _a_weighting(self, freqs: np.ndarray) -> np.ndarray:
        """Apply A-weighting filter"""
        f = freqs
        f2 = f**2
        A1000 = 7.39705e9
        
        numerator = A1000 * f2**2
        denominator = ((f2 + 20.6**2) * 
                      np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2)) * 
                      (f2 + 12194**2))
        
        return numerator / (denominator + 1e-10)


class HighAccuracyFeatureEngineer:
    """Advanced feature engineering for maximum predictive power"""
    
    def __init__(self):
        self.polynomial_features = None
        self.feature_selector = None
        self.scaler = None
        
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer advanced features for high accuracy"""
        df_eng = df.copy()
        
        # 1. Interaction features
        df_eng = self._create_interaction_features(df_eng)
        
        # 2. Temporal features (advanced)
        df_eng = self._create_advanced_temporal_features(df_eng)
        
        # 3. Aggregation features
        df_eng = self._create_aggregation_features(df_eng)
        
        # 4. Ratio features
        df_eng = self._create_ratio_features(df_eng)
        
        # 5. Binning features
        df_eng = self._create_binning_features(df_eng)
        
        return df_eng
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between key variables"""
        # Noise × Time interactions
        if 'spl_rms' in df.columns and 'hour' in df.columns:
            df['spl_hour_interaction'] = df['spl_rms'] * df['hour']
        
        if 'spl_rms' in df.columns and 'is_rush_hour' in df.columns:
            df['spl_rush_interaction'] = df['spl_rms'] * df['is_rush_hour']
        
        # Age × Noise interactions
        if 'age' in df.columns and 'spl_rms' in df.columns:
            df['age_noise_interaction'] = df['age'] * df['spl_rms']
            df['age_squared'] = df['age'] ** 2
        
        # Spectral feature interactions
        if 'spectral_centroid_mean' in df.columns and 'spectral_bandwidth_mean' in df.columns:
            df['spectral_centroid_bandwidth_ratio'] = (df['spectral_centroid_mean'] / 
                                                      (df['spectral_bandwidth_mean'] + 1e-10))
        
        # Psychoacoustic interactions
        if 'roughness' in df.columns and 'sharpness' in df.columns:
            df['roughness_sharpness_product'] = df['roughness'] * df['sharpness']
        
        return df
    
    def _create_advanced_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced temporal features"""
        if 'hour' in df.columns:
            # Cyclical encoding for hour
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        if 'day_of_week' in df.columns:
            # Cyclical encoding for day of week
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        if 'month' in df.columns:
            # Cyclical encoding for month
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def _create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features"""
        # MFCC aggregations
        mfcc_cols = [col for col in df.columns if 'mfcc_' in col and '_mean' in col][:10]
        if len(mfcc_cols) > 0:
            df['mfcc_mean_avg'] = df[mfcc_cols].mean(axis=1)
            df['mfcc_mean_std'] = df[mfcc_cols].std(axis=1)
            df['mfcc_mean_max'] = df[mfcc_cols].max(axis=1)
            df['mfcc_mean_min'] = df[mfcc_cols].min(axis=1)
        
        # Frequency band aggregations
        band_cols = [col for col in df.columns if '_energy_mean' in col]
        if len(band_cols) > 0:
            df['total_band_energy'] = df[band_cols].sum(axis=1)
            df['dominant_band_energy'] = df[band_cols].max(axis=1)
            df['energy_concentration'] = df['dominant_band_energy'] / (df['total_band_energy'] + 1e-10)
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features"""
        # SPL ratios
        if 'spl_peak' in df.columns and 'spl_rms' in df.columns:
            df['peak_to_rms_ratio'] = df['spl_peak'] / (df['spl_rms'] + 1e-10)
        
        if 'spl_percentile_90' in df.columns and 'spl_percentile_10' in df.columns:
            df['spl_range_ratio'] = (df['spl_percentile_90'] / 
                                    (df['spl_percentile_10'] + 1e-10))
        
        # Energy ratios
        if 'harmonic_energy' in df.columns and 'percussive_energy' in df.columns:
            total_hp_energy = df['harmonic_energy'] + df['percussive_energy']
            df['harmonic_ratio'] = df['harmonic_energy'] / (total_hp_energy + 1e-10)
            df['percussive_ratio'] = df['percussive_energy'] / (total_hp_energy + 1e-10)
        
        return df
    
    def _create_binning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binning features for non-linear relationships"""
        # Age groups (more granular)
        if 'age' in df.columns:
            df['age_group_fine'] = pd.cut(df['age'], 
                                         bins=[0, 25, 30, 35, 40, 50, 60, 100],
                                         labels=['very_young', 'young', 'early_career', 
                                                'mid_career', 'mature', 'senior', 'elderly'])
        
        # Noise level categories
        if 'spl_rms' in df.columns:
            df['noise_category'] = pd.cut(df['spl_rms'],
                                         bins=[0, 40, 50, 60, 70, 80, 100],
                                         labels=['very_quiet', 'quiet', 'moderate', 
                                                'loud', 'very_loud', 'extreme'])
        
        return df


class AdvancedStressPredictionModels:
    """Advanced ML models with hyperparameter optimization for maximum accuracy"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.models = {}
        self.best_model = None
        self.feature_engineer = HighAccuracyFeatureEngineer()
        self.results = {}
        logger.info(f"Advanced models initialized with {len(df)} records")
    
    def prepare_high_quality_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare high-quality feature set for maximum accuracy"""
        try:
            # Start with engineered features
            df_eng = self.feature_engineer.engineer_features(self.df)
            
            # Define comprehensive feature sets
            acoustic_features = [
                'spl_rms', 'spl_peak', 'spl_percentile_90', 'spl_percentile_50', 'spl_percentile_10',
                'dynamic_range', 'crest_factor', 'energy_variance', 'energy_mean',
                'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_skew',
                'spectral_bandwidth_mean', 'spectral_bandwidth_std',
                'spectral_rolloff_85_mean', 'spectral_rolloff_90_mean', 'spectral_rolloff_95_mean',
                'zcr_mean', 'zcr_std', 'zcr_max', 'zcr_min',
                'spectral_flatness_mean', 'spectral_flatness_std',
                'roughness', 'sharpness', 'loudness', 'tonality',
                'harmonic_energy', 'percussive_energy', 'harmonic_percussive_ratio'
            ]
            
            # Enhanced MFCC features
            mfcc_features = []
            for i in range(min(13, config.N_MFCC)):  # Use first 13 MFCCs
                mfcc_features.extend([
                    f'mfcc_{i}_mean', f'mfcc_{i}_std', f'mfcc_{i}_max', f'mfcc_{i}_min'
                ])
            
            # Delta MFCC features
            for i in range(5):  # First 5 delta MFCCs
                mfcc_features.extend([f'mfcc_delta_{i}_mean', f'mfcc_delta_{i}_std'])
            
            # Frequency band features
            freq_band_features = []
            for band in ['sub_bass', 'bass', 'low_mid', 'mid', 'upper_mid', 'presence', 'brilliance']:
                freq_band_features.extend([
                    f'{band}_energy_mean', f'{band}_energy_std', f'{band}_energy_max'
                ])
            
            # Temporal features
            temporal_features = [
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
                'is_rush_hour', 'is_weekend', 'onset_rate', 'tempo', 'beat_regularity'
            ]
            
            # Interaction and engineered features
            engineered_features = [
                'spl_hour_interaction', 'spl_rush_interaction', 'age_noise_interaction',
                'spectral_centroid_bandwidth_ratio', 'roughness_sharpness_product',
                'peak_to_rms_ratio', 'spl_range_ratio', 'harmonic_ratio',
                'mfcc_mean_avg', 'total_band_energy', 'energy_concentration'
            ]
            
            # Demographic and environmental
            context_features = [
                'age', 'age_squared', 'temperature', 'humidity', 'wind_speed',
                'population_density', 'daily_traffic_count', 'cortisol_level',
                'heart_rate_variability'
            ]
            
            # Combine all feature groups
            all_features = (acoustic_features + mfcc_features + freq_band_features + 
                           temporal_features + engineered_features + context_features)
            
            # Select available features
            available_features = [f for f in all_features if f in df_eng.columns]
            logger.info(f"Selected {len(available_features)} features from {len(all_features)} possible")
            
            # Handle categorical variables
            categorical_cols = []
            if 'gender' in df_eng.columns:
                gender_dummies = pd.get_dummies(df_eng['gender'], prefix='gender', drop_first=True)
                df_eng = pd.concat([df_eng, gender_dummies], axis=1)
                available_features.extend(gender_dummies.columns.tolist())
                categorical_cols.extend(gender_dummies.columns.tolist())
            
            if 'noise_category' in df_eng.columns:
                noise_dummies = pd.get_dummies(df_eng['noise_category'], prefix='noise_cat', drop_first=True)
                df_eng = pd.concat([df_eng, noise_dummies], axis=1)
                available_features.extend(noise_dummies.columns.tolist())
                categorical_cols.extend(noise_dummies.columns.tolist())
            
            # Create final feature matrix
            X = df_eng[available_features].copy()
            
            # Handle missing values strategically
            numeric_cols = [col for col in X.columns if col not in categorical_cols]
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
            X[categorical_cols] = X[categorical_cols].fillna(0)  # For dummy variables
            
            # Target variable
            y = df_eng['composite_stress_score'].copy()
            
            # Remove any remaining NaN values in target
            valid_indices = ~y.isna()
            X = X[valid_indices]
            y = y[valid_indices]
            
            self.feature_names = available_features
            logger.info(f"Final feature matrix: {X.shape}, Target: {y.shape}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame(), pd.Series()
    
    def train_optimized_models(self) -> Dict[str, Dict[str, float]]:
        """Train models with hyperparameter optimization"""
        try:
            X, y = self.prepare_high_quality_features()
            
            if X.empty or len(y) == 0:
                logger.error("No valid data for training")
                return {}
            
            # Train-test split with stratification based on stress levels
            y_binned = pd.cut(y, bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE,
                stratify=y_binned
            )
            
            # Store test data for later use
            self.X_test = X_test
            self.y_test = y_test
            
            results = {}
            
            # 1. Optimized Random Forest
            logger.info("Training optimized Random Forest...")
            rf_model = self._train_optimized_random_forest(X_train, y_train, X_test, y_test)
            results['optimized_rf'] = rf_model['metrics']
            self.models['optimized_rf'] = rf_model['model']
            
            # 2. Optimized XGBoost
            logger.info("Training optimized XGBoost...")
            xgb_model = self._train_optimized_xgboost(X_train, y_train, X_test, y_test)
            results['optimized_xgb'] = xgb_model['metrics']
            self.models['optimized_xgb'] = xgb_model['model']
            
            # 3. Optimized LightGBM (if available)
            if LIGHTGBM_AVAILABLE:
                logger.info("Training optimized LightGBM...")
                lgb_model = self._train_optimized_lightgbm(X_train, y_train, X_test, y_test)
                results['optimized_lgb'] = lgb_model['metrics']
                self.models['optimized_lgb'] = lgb_model['model']
            
            # 4. Optimized CatBoost (if available)
            if CATBOOST_AVAILABLE:
                logger.info("Training optimized CatBoost...")
                cat_model = self._train_optimized_catboost(X_train, y_train, X_test, y_test)
                results['optimized_catboost'] = cat_model['metrics']
                self.models['optimized_catboost'] = cat_model['model']
            
            # 5. Advanced Neural Network
            if TF_AVAILABLE:
                logger.info("Training advanced neural network...")
                nn_model = self._train_advanced_neural_network(X_train, y_train, X_test, y_test)
                results['advanced_nn'] = nn_model['metrics']
                self.models['advanced_nn'] = nn_model['model']
            
            # 6. Ensemble methods
            logger.info("Creating ensemble models...")
            ensemble_results = self._create_advanced_ensemble(X_train, y_train, X_test, y_test)
            results.update(ensemble_results)
            
            # Find best model
            best_model_name = max(results.keys(), key=lambda k: results[k].get('r2', 0))
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            
            logger.info(f"Best model: {best_model_name} with R² = {results[best_model_name]['r2']:.4f}")
            
            self.results = results
            return results
            
        except Exception as e:
            logger.error(f"Error training optimized models: {e}")
            return {}
    
    def _train_optimized_random_forest(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train Random Forest with hyperparameter optimization"""
        param_grid = {
            'n_estimators': [200, 500, 800],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', 0.8],
            'bootstrap': [True, False]
        }
        
        rf = RandomForestRegressor(random_state=config.RANDOM_STATE, n_jobs=-1)
        
        # Use RandomizedSearchCV for efficiency
        rf_random = RandomizedSearchCV(
            rf, param_grid, n_iter=50, cv=5, 
            scoring='r2', random_state=config.RANDOM_STATE, n_jobs=-1
        )
        
        rf_random.fit(X_train, y_train)
        best_rf = rf_random.best_estimator_
        
        y_pred = best_rf.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return {'model': best_rf, 'metrics': metrics}
    
    def _train_optimized_xgboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train XGBoost with hyperparameter optimization"""
        param_grid = {
            'n_estimators': [200, 500, 800],
            'max_depth': [6, 10, 15],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [0, 0.1, 1]
        }
        
        xgb_model = xgb.XGBRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=0)
        
        xgb_random = RandomizedSearchCV(
            xgb_model, param_grid, n_iter=50, cv=5,
            scoring='r2', random_state=config.RANDOM_STATE, n_jobs=-1
        )
        
        xgb_random.fit(X_train, y_train)
        best_xgb = xgb_random.best_estimator_
        
        y_pred = best_xgb.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return {'model': best_xgb, 'metrics': metrics}
    
    def _train_optimized_lightgbm(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train LightGBM with hyperparameter optimization"""
        param_grid = {
            'n_estimators': [200, 500, 800],
            'max_depth': [10, 20, -1],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 50, 100],
            'min_data_in_leaf': [10, 20, 50],
            'feature_fraction': [0.8, 0.9, 1.0],
            'bagging_fraction': [0.8, 0.9, 1.0]
        }
        
        lgb_model = lgb.LGBMRegressor(random_state=config.RANDOM_STATE, n_jobs=-1, verbosity=-1)
        
        lgb_random = RandomizedSearchCV(
            lgb_model, param_grid, n_iter=50, cv=5,
            scoring='r2', random_state=config.RANDOM_STATE, n_jobs=-1
        )
        
        lgb_random.fit(X_train, y_train)
        best_lgb = lgb_random.best_estimator_
        
        y_pred = best_lgb.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return {'model': best_lgb, 'metrics': metrics}
    
    def _train_optimized_catboost(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train CatBoost with hyperparameter optimization"""
        param_grid = {
            'iterations': [200, 500, 800],
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128]
        }
        
        cat_model = CatBoostRegressor(
            random_state=config.RANDOM_STATE, 
            verbose=False, 
            allow_writing_files=False
        )
        
        cat_random = RandomizedSearchCV(
            cat_model, param_grid, n_iter=30, cv=5,  # Reduced iterations for CatBoost
            scoring='r2', random_state=config.RANDOM_STATE, n_jobs=1  # CatBoost handles parallelism internally
        )
        
        cat_random.fit(X_train, y_train)
        best_cat = cat_random.best_estimator_
        
        y_pred = best_cat.predict(X_test)
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        return {'model': best_cat, 'metrics': metrics}
    
    def _train_advanced_neural_network(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train advanced neural network with optimization"""
        # Scale features for neural network
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build advanced architecture
        input_dim = X_train_scaled.shape[1]
        
        inputs = Input(shape=(input_dim,))
        
        # First hidden layer with dropout and batch norm
        x = Dense(256, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second hidden layer
        x = Dense(128, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Third hidden layer
        x = Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Fourth hidden layer
        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)
        
        # Output layer
        outputs = Dense(1, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Advanced optimizer
        optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Advanced callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            ModelCheckpoint('temp_best_model.h5', monitor='val_loss', save_best_only=True)
        ]
        
        # Train with validation split
        history = model.fit(
            X_train_scaled, y_train,
            epochs=200,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # Predict
        y_pred = model.predict(X_test_scaled, verbose=0).flatten()
        
        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
        
        # Store scaler with model
        model.scaler = scaler
        
        return {'model': model, 'metrics': metrics}
    
    def _create_advanced_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        """Create advanced ensemble methods"""
        ensemble_results = {}
        
        # 1. Voting Regressor (if we have multiple models)
        if len(self.models) >= 2:
            # Get the best models for ensemble
            model_items = list(self.models.items())
            
            voting_estimators = []
            for name, model in model_items:
                if hasattr(model, 'predict'):  # Skip if model is not trained properly
                    voting_estimators.append((name, model))
            
            if len(voting_estimators) >= 2:
                voting_reg = VotingRegressor(estimators=voting_estimators)
                voting_reg.fit(X_train, y_train)
                
                y_pred = voting_reg.predict(X_test)
                
                ensemble_results['voting_ensemble'] = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred)
                }
                
                self.models['voting_ensemble'] = voting_reg
        
        # 2. Weighted Average Ensemble
        if len(self.models) >= 2:
            predictions = []
            weights = []
            
            for name, model in self.models.items():
                if name != 'voting_ensemble' and hasattr(model, 'predict'):
                    try:
                        if hasattr(model, 'scaler'):  # Neural network
                            pred = model.predict(model.scaler.transform(X_test), verbose=0).flatten()
                        else:
                            pred = model.predict(X_test)
                        
                        predictions.append(pred)
                        
                        # Weight by R² score (higher R² gets higher weight)
                        model_r2 = r2_score(y_test, pred)
                        weights.append(max(0, model_r2))  # Ensure non-negative weights
                        
                    except Exception as e:
                        logger.warning(f"Could not get predictions from {name}: {e}")
                        continue
            
            if len(predictions) >= 2 and sum(weights) > 0:
                # Normalize weights
                weights = np.array(weights) / sum(weights)
                
                # Weighted average
                weighted_pred = np.average(predictions, axis=0, weights=weights)
                
                ensemble_results['weighted_ensemble'] = {
                    'r2': r2_score(y_test, weighted_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, weighted_pred)),
                    'mae': mean_absolute_error(y_test, weighted_pred)
                }
        
        return ensemble_results
