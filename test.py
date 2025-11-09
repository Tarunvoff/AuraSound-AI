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