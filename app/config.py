import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    """Configuration settings for the Noise-Mental Health Analytics app"""
    
    # App settings
    APP_TITLE = "Noise-Mental Health Analytics"
    APP_SUBTITLE = "Advanced Analytics for Environmental Noise Impact on Mental Wellbeing"
    APP_EMOJI = "🎧"
    
    # File paths
    DATA_DIR = Path("data")
    MODELS_DIR = Path("models") 
    RESULTS_DIR = Path("results")
    LOGS_DIR = Path("logs")
    
    # Model parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 10
    
    # Audio processing
    SAMPLE_RATE = 44100
    AUDIO_DURATION = 10
    N_MFCC = 20
    N_CHROMA = 12
    N_MEL = 128
    
    # UI Theme - Premium Dark Mode
    PRIMARY_COLOR = "#00f5ff"    # Neon cyan
    SECONDARY_COLOR = "#007bff"  # Electric blue
    ACCENT_COLOR = "#5b46ff"     # Deep purple
    BACKGROUND_COLOR = "#0a0f1c" # Deep navy
    SURFACE_COLOR = "#111827"    # Dark slate
    TEXT_COLOR = "#eaf2ff"       # Soft white
    
    # Glass effect settings
    GLASS_BACKGROUND = "rgba(255, 255, 255, 0.06)"
    GLASS_BORDER = "rgba(255, 255, 255, 0.1)"
    GLASS_BLUR = "blur(10px)"
    
    # Hero section
    HERO_BACKGROUND = "https://images.unsplash.com/photo-1614624532983-4ce03382d63d?auto=format&fit=crop&w=2000"
    HERO_TITLE = "Environmental Noise Analytics"
    HERO_SUBTITLE = "Discover the Impact of Urban Soundscapes on Mental Wellbeing"
    
    def __post_init__(self):
        # Create necessary directories
        for dir_path in [self.DATA_DIR, self.MODELS_DIR, self.RESULTS_DIR, self.LOGS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

# Global config instance
config = Config()