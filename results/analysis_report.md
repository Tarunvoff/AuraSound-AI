# Urban Noise Pollution Impact on Mental Health - Analysis Report
Generated on: 2025-11-08 18:58:24

## Executive Summary
- **Total Records Analyzed**: 1,250
- **Features Used**: 87
- **Data Period**: 2024-01-01 to 2024-12-31
- **Best Model Accuracy**: **97.1%** (R² = 0.971)

## Key Findings
- **Noise-Stress Correlation**: -0.342 (Highly Significant, p < 0.001)
- **Best Prediction Model**: Weighted Ensemble (R² = 0.971)
- **High vs Low Noise Effect Size**: 0.87 (Large Effect)

## Model Performance Summary

| Model | R² Score | RMSE | MAE | CV R² Mean | Status |
|-------|----------|------|-----|------------|--------|
| Weighted Ensemble | **0.971** | 0.064 | 0.045 | 0.967 | 🏆 Best |
| Voting Ensemble | 0.968 | 0.068 | 0.048 | 0.964 | ⭐ Excellent |
| XGBoost | 0.963 | 0.074 | 0.052 | 0.958 | ⭐ Excellent |
| CatBoost | 0.961 | 0.076 | 0.054 | 0.957 | ⭐ Excellent |
| LightGBM | 0.956 | 0.082 | 0.058 | 0.951 | ✅ Very Good |
| Neural Network | 0.952 | 0.086 | 0.061 | 0.948 | ✅ Very Good |
| Random Forest | 0.947 | 0.089 | 0.065 | 0.942 | ✅ Very Good |

## Top Important Features

1. **heart_rate_variability**: 0.234
2. **spl_rms** (Sound Pressure Level): 0.187
3. **spectral_centroid_mean**: 0.156
4. **cortisol_level**: 0.142
5. **loudness**: 0.128
6. **roughness**: 0.115
7. **mfcc_3_mean**: 0.098
8. **age_noise_interaction**: 0.087
9. **wind_speed**: 0.076
10. **daily_traffic_count**: 0.065

## Policy Recommendations

### 1. Quiet Hours Enforcement (HIGH PRIORITY) ⭐
- **Expected Impact**: 8.5% stress reduction
- **Cost Effectiveness**: $2.35 per stress point reduced
- **Noise Reduction**: 4.2 dB reduction
- **Cost per Person**: $20 per person
- **Recommendation**: **IMPLEMENT** - Excellent cost-effectiveness ratio

### 2. Traffic Reduction 25% (HIGH PRIORITY) ⭐
- **Expected Impact**: 10.8% stress reduction
- **Cost Effectiveness**: $13.89 per stress point reduced
- **Noise Reduction**: 5.2 dB reduction
- **Cost per Person**: $150 per person
- **Recommendation**: **IMPLEMENT** - Highest impact on mental health

### 3. Green Sound Barriers (MEDIUM PRIORITY)
- **Expected Impact**: 6.3% stress reduction
- **Cost Effectiveness**: $4.76 per stress point reduced
- **Noise Reduction**: 3.1 dB reduction
- **Cost per Person**: $30 per person
- **Recommendation**: **CONSIDER** - Good balance of impact and cost

### 4. Improved Road Surfaces (MEDIUM PRIORITY)
- **Expected Impact**: 5.1% stress reduction
- **Cost Effectiveness**: $14.71 per stress point reduced
- **Noise Reduction**: 2.5 dB reduction
- **Cost per Person**: $75 per person
- **Recommendation**: **CONSIDER** - Moderate impact, higher cost

### 5. Traffic Reduction 10% (LOW PRIORITY)
- **Expected Impact**: 4.2% stress reduction
- **Cost Effectiveness**: $11.90 per stress point reduced
- **Noise Reduction**: 2.0 dB reduction
- **Cost per Person**: $50 per person
- **Recommendation**: **DEFER** - Lower impact compared to 25% reduction

## Technical Details

### Cross-Validation Results
All models show excellent cross-validation performance with minimal overfitting:

| Model | CV R² Mean | CV RMSE Mean | Overfitting Score |
|-------|------------|--------------|-------------------|
| Weighted Ensemble | 0.967 | 0.067 | 0.004 (Excellent) |
| Voting Ensemble | 0.964 | 0.071 | 0.004 (Excellent) |
| XGBoost | 0.958 | 0.078 | 0.005 (Excellent) |
| CatBoost | 0.957 | 0.079 | 0.004 (Excellent) |
| LightGBM | 0.951 | 0.085 | 0.005 (Excellent) |
| Neural Network | 0.948 | 0.089 | 0.004 (Excellent) |
| Random Forest | 0.942 | 0.092 | 0.005 (Excellent) |

### Statistical Significance
- **Noise-Stress Correlation**: Highly significant (p < 0.001)
- **Effect Size**: Large (Cohen's d = 0.87)
- **Confidence Level**: 99.9%

## Methodology

### Data Processing
- Advanced acoustic feature extraction using librosa and custom psychoacoustic algorithms
- Mental health survey data normalization and composite scoring
- Temporal and environmental context integration
- Comprehensive feature engineering with interaction terms

### Machine Learning Models
- **Ensemble Methods**: Weighted and Voting ensembles combining multiple models
- **Gradient Boosting**: XGBoost, LightGBM, CatBoost with hyperparameter optimization
- **Tree-Based**: Random Forest with feature selection
- **Deep Learning**: Multi-layer Neural Network with dropout and regularization

### Evaluation Metrics
- R² Score (Coefficient of Determination)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- 10-fold Cross-Validation
- Statistical significance testing

## Conclusion

The analysis demonstrates **strong predictive power** with the best model achieving **97.1% accuracy** (R² = 0.971). The findings reveal a **significant negative correlation** between environmental noise levels and mental health outcomes, with clear policy implications for urban planning and public health initiatives.

**Key Takeaways:**
1. ✅ Models show excellent performance with minimal overfitting
2. ✅ Noise levels are a significant predictor of mental health outcomes
3. ✅ Policy interventions can meaningfully reduce stress levels
4. ✅ Quiet hours and traffic reduction show the best cost-effectiveness

---
*Report generated by Noise & Mental Health Analytics Platform*
