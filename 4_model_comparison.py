"""
4_model_comparison.py
STEP 4: Compare Linear Regression vs Prophet Time-Series Model
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("=" * 60)
print("STEP 4: MODEL COMPARISON")
print("=" * 60)

# Load predictions from both models
print("\n📂 Loading predictions from both models...")

try:
    lr_predictions = pd.read_csv('linear_regression_predictions.csv')
    lr_predictions['date'] = pd.to_datetime(lr_predictions['date'])
    print("✅ Linear Regression predictions loaded")
except FileNotFoundError:
    print("❌ Run 2_linear_regression.py first!")
    exit()

try:
    prophet_predictions = pd.read_csv('prophet_test_predictions.csv')
    prophet_predictions['date'] = pd.to_datetime(prophet_predictions['date'])
    print("✅ Prophet predictions loaded")
except FileNotFoundError:
    print("❌ Run 3_time_series.py first!")
    exit()

# Calculate metrics for both models
print("\n1️⃣ CALCULATING PERFORMANCE METRICS")
print("-" * 40)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Linear Regression metrics
lr_mae = mean_absolute_error(lr_predictions['actual_sales'], 
                              lr_predictions['predicted_sales'])
lr_rmse = np.sqrt(mean_squared_error(lr_predictions['actual_sales'], 
                                      lr_predictions['predicted_sales']))
lr_r2 = r2_score(lr_predictions['actual_sales'], 
                 lr_predictions['predicted_sales'])
lr_mape = np.mean(np.abs(lr_predictions['error_percentage']))

# Prophet metrics
prophet_mae = mean_absolute_error(prophet_predictions['actual_sales'], 
                                   prophet_predictions['predicted_sales'])
prophet_rmse = np.sqrt(mean_squared_error(prophet_predictions['actual_sales'], 
                                           prophet_predictions['predicted_sales']))
prophet_r2 = r2_score(prophet_predictions['actual_sales'], 
                      prophet_predictions['predicted_sales'])
prophet_mape = np.mean(np.abs(prophet_predictions['error'] / 
                               prophet_predictions['actual_sales'] * 100))

# Create comparison dataframe
comparison = pd.DataFrame({
    'Metric': ['MAE ($)', 'RMSE ($)', 'R² Score', 'MAPE (%)'],
    'Linear Regression': [lr_mae, lr_rmse, lr_r2, lr_mape],
    'Prophet (Time-Series)': [prophet_mae, prophet_rmse, prophet_r2, prophet_mape]
})

print("\n📊 PERFORMANCE COMPARISON")
print("=" * 60)
print(comparison.to_string(index=False))

# Determine winner
print("\n🏆 WINNER BY METRIC:")
print("-" * 40)
for idx, row in comparison.iterrows():
    metric = row['Metric']
    lr_val = row['Linear Regression']
    prophet_val = row['Prophet (Time-Series)']
    
    if metric == 'R² Score':
        winner = 'Linear Regression' if lr_val > prophet_val else 'Prophet'
    else:
        winner = 'Linear Regression' if lr_val < prophet_val else 'Prophet'
    
    print(f"{metric}: {winner}")

# Overall recommendation
lr_wins = sum([
    lr_mae < prophet_mae,
    lr_rmse < prophet_rmse,
    lr_r2 > prophet_r2,
    lr_mape < prophet_mape
])

print("\n🎯 OVERALL RECOMMENDATION:")
print("-" * 40)
if lr_wins >= 3:
    print("✅ Linear Regression is the better model for this dataset")
    best_model = "Linear Regression"
else:
    print("✅ Prophet (Time-Series) is the better model for this dataset")
    best_model = "Prophet"

# Visualizations
print("\n2️⃣ CREATING COMPARISON VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Side-by-side predictions
dates = lr_predictions['date']
actuals = lr_predictions['actual_sales']

axes[0, 0].plot(dates, actuals, label='Actual', marker='o', linewidth=2.5, color='black')
axes[0, 0].plot(dates, lr_predictions['predicted_sales'], 
                label='Linear Regression', marker='s', linewidth=2, alpha=0.7)
axes[0, 0].plot(dates, prophet_predictions['predicted_sales'], 
                label='Prophet', marker='^', linewidth=2, alpha=0.7)
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].set_title('Model Predictions Comparison', fontweight='bold', fontsize=14)
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. Metric comparison bar chart
metrics_plot = comparison.set_index('Metric')
x = np.arange(len(metrics_plot))
width = 0.35

bars1 = axes[0, 1].bar(x - width/2, metrics_plot['Linear Regression'], 
                       width, label='Linear Regression', alpha=0.8)
bars2 = axes[0, 1].bar(x + width/2, metrics_plot['Prophet (Time-Series)'], 
                       width, label='Prophet', alpha=0.8)

axes[0, 1].set_xlabel('Metric')
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_title('Performance Metrics Comparison', fontweight='bold', fontsize=14)
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(metrics_plot.index, rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 3. Error distribution comparison
axes[1, 0].hist(lr_predictions['error'], bins=20, alpha=0.6, 
                label='Linear Regression', edgecolor='black')
axes[1, 0].hist(prophet_predictions['error'], bins=20, alpha=0.6, 
                label='Prophet', edgecolor='black')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Prediction Error ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Error Distribution Comparison', fontweight='bold', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Absolute error over time
axes[1, 1].plot(dates, np.abs(lr_predictions['error']), 
                label='Linear Regression', marker='o', linewidth=2)
axes[1, 1].plot(dates, np.abs(prophet_predictions['error']), 
                label='Prophet', marker='s', linewidth=2)
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Absolute Error ($)')
axes[1, 1].set_title('Absolute Error Over Time', fontweight='bold', fontsize=14)
axes[1, 1].legend()
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('4_model_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Comparison visualization saved: 4_model_comparison.png")

# Summary statistics
print("\n3️⃣ DETAILED STATISTICS")
print("-" * 40)

print("\nLinear Regression:")
print(f"  • Mean error: ${lr_predictions['error'].mean():,.2f}")
print(f"  • Std dev of errors: ${lr_predictions['error'].std():,.2f}")
print(f"  • Max overestimation: ${lr_predictions['error'].max():,.2f}")
print(f"  • Max underestimation: ${lr_predictions['error'].min():,.2f}")

print("\nProphet:")
print(f"  • Mean error: ${prophet_predictions['error'].mean():,.2f}")
print(f"  • Std dev of errors: ${prophet_predictions['error'].std():,.2f}")
print(f"  • Max overestimation: ${prophet_predictions['error'].max():,.2f}")
print(f"  • Max underestimation: ${prophet_predictions['error'].min():,.2f}")

# Save comparison report
print("\n4️⃣ SAVING COMPARISON REPORT")
print("-" * 40)

report = f"""
SALES PREDICTION MODEL COMPARISON REPORT
=========================================

Dataset Information:
- Total test samples: {len(lr_predictions)}
- Date range: {dates.min()} to {dates.max()}
- Actual sales range: ${actuals.min():,.0f} - ${actuals.max():,.0f}

Performance Metrics:
-------------------
{comparison.to_string(index=False)}

Winner by Metric:
-----------------
MAE: {"Linear Regression" if lr_mae < prophet_mae else "Prophet"}
RMSE: {"Linear Regression" if lr_rmse < prophet_rmse else "Prophet"}
R²: {"Linear Regression" if lr_r2 > prophet_r2 else "Prophet"}
MAPE: {"Linear Regression" if lr_mape < prophet_mape else "Prophet"}

Overall Recommendation:
----------------------
{best_model} is the better model for this dataset.

Model Characteristics:

Linear Regression:
+ Better for: Datasets with clear relationships between features and sales
+ Pros: Fast training, interpretable coefficients, works well with multiple features
+ Cons: Assumes linear relationships, may not capture complex patterns

Prophet (Time-Series):
+ Better for: Datasets with strong temporal patterns and seasonality
+ Pros: Handles trends and seasonality automatically, provides uncertainty intervals
+ Cons: Requires time-ordered data, less flexible with additional features

When to Use Each Model:
-----------------------
Use Linear Regression if:
- You have many relevant features (promotions, weather, competitors)
- Relationships between features and sales are relatively stable
- You need fast predictions
- Interpretability is important

Use Prophet if:
- Time-based patterns (seasonality, trends) are dominant
- You need confidence intervals
- You have limited additional features
- Historical temporal patterns are expected to continue
"""

with open('model_comparison_report.txt', 'w') as f:
    f.write(report)

print("✅ Report saved: model_comparison_report.txt")

print("\n" + "=" * 60)
print("✅ MODEL COMPARISON COMPLETE!")
print("=" * 60)
print(f"\n🎯 Recommended model: {best_model}")
print(f"\n📊 Key metrics:")
print(f"  • MAE: ${min(lr_mae, prophet_mae):,.2f}")
print(f"  • R²: {max(lr_r2, prophet_r2):.4f}")
print("\n📁 All results saved to files")
print("📈 Check the visualizations and report for detailed analysis")
