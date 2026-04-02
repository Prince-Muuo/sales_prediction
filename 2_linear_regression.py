import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("STEP 2: LINEAR REGRESSION MODEL")
print("=" * 60)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Feature engineering: extracting time-based patterns from date
# These features help the model understand seasonality and trends
print("\n1️⃣ FEATURE ENGINEERING")
print("-" * 40)
df['day_of_week'] = df['date'].dt.dayofweek
df['week_of_year'] = df['date'].dt.isocalendar().week
df['month'] = df['date'].dt.month
df['quarter'] = df['date'].dt.quarter
df['year'] = df['date'].dt.year
df['days_since_start'] = (df['date'] - df['date'].min()).dt.days

print("Features created:")
print(f"  • week_of_year: Week number (1-52)")
print(f"  • month: Month number (1-12)")
print(f"  • quarter: Quarter (1-4)")
print(f"  • year: Year")
print(f"  • days_since_start: Days since first record (captures trend)")

# Select features for the model
features = [
    'promotion',
    'holiday',
    'week_of_year',
    'month',
    'quarter',
    'year',
    'days_since_start',
    'day_of_week'
]

X = df[features]
y = df['sales']

print(f"\n📊 Feature matrix shape: {X.shape}")
print(f"📊 Target vector shape: {y.shape}")

# Split data: 80% training, 20% testing
print("\n2️⃣ SPLITTING DATA")
print("-" * 40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False  # shuffle=False preserves time order
)

print(f"Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
print(f"Test set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

# Optional: Feature scaling (can improve model performance)
print("\n3️⃣ FEATURE SCALING")
print("-" * 40)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled using StandardScaler")

# Train the model
print("\n4️⃣ TRAINING LINEAR REGRESSION MODEL")
print("-" * 40)
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("✅ Model trained successfully!")

# Feature importance (coefficients)
print("\n5️⃣ FEATURE IMPORTANCE")
print("-" * 40)
feature_importance = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_
}).sort_values('coefficient', key=abs, ascending=False)

print(feature_importance)

# Make predictions
print("\n6️⃣ MAKING PREDICTIONS")
print("-" * 40)
train_predictions = model.predict(X_train_scaled)
test_predictions = model.predict(X_test_scaled)
print(f"✅ Generated {len(test_predictions)} predictions for test set")

# Evaluate the model
print("\n7️⃣ MODEL PERFORMANCE")
print("-" * 40)

# Training metrics
train_mae = mean_absolute_error(y_train, train_predictions)
train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
train_r2 = r2_score(y_train, train_predictions)

print("TRAINING SET:")
print(f"  • MAE (Mean Absolute Error): ${train_mae:,.2f}")
print(f"  • RMSE (Root Mean Squared Error): ${train_rmse:,.2f}")
print(f"  • R² Score: {train_r2:.4f}")

# Test metrics
test_mae = mean_absolute_error(y_test, test_predictions)
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
test_r2 = r2_score(y_test, test_predictions)

print("\nTEST SET:")
print(f"  • MAE (Mean Absolute Error): ${test_mae:,.2f}")
print(f"  • RMSE (Root Mean Squared Error): ${test_rmse:,.2f}")
print(f"  • R² Score: {test_r2:.4f}")

print("\n💡 What do these metrics mean?")
print(f"  • MAE: On average, predictions are off by ${test_mae:,.2f}")
print(f"  • RMSE: Penalizes larger errors more than MAE")
print(f"  • R²: Model explains {test_r2*100:.1f}% of sales variation")

# Visualizations
print("\n8️⃣ CREATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Actual vs Predicted (Test Set)
axes[0, 0].scatter(y_test, test_predictions, alpha=0.6)
axes[0, 0].plot([y_test.min(), y_test.max()], 
                [y_test.min(), y_test.max()], 
                'r--', lw=2, label='Perfect Prediction')
axes[0, 0].set_xlabel('Actual Sales ($)')
axes[0, 0].set_ylabel('Predicted Sales ($)')
axes[0, 0].set_title('Actual vs Predicted Sales (Test Set)', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals (errors)
residuals = y_test.values - test_predictions
axes[0, 1].scatter(test_predictions, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
axes[0, 1].set_xlabel('Predicted Sales ($)')
axes[0, 1].set_ylabel('Residuals ($)')
axes[0, 1].set_title('Residual Plot (Test Set)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 3. Predictions over time
test_dates = df['date'].iloc[-len(y_test):].values
axes[1, 0].plot(test_dates, y_test.values, label='Actual', marker='o', linewidth=2)
axes[1, 0].plot(test_dates, test_predictions, label='Predicted', 
                marker='s', linewidth=2, alpha=0.7)
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Sales ($)')
axes[1, 0].set_title('Sales Predictions Over Time (Test Set)', fontweight='bold')
axes[1, 0].legend()
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(True, alpha=0.3)

# 4. Feature importance
top_features = feature_importance.head(8)
axes[1, 1].barh(top_features['feature'], abs(top_features['coefficient']))
axes[1, 1].set_xlabel('Absolute Coefficient Value')
axes[1, 1].set_title('Top Feature Importance', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('2_linear_regression_results.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: 2_linear_regression_results.png")

# Save predictions to CSV
print("\n9️⃣ SAVING RESULTS")
print("-" * 40)
results_df = pd.DataFrame({
    'date': test_dates,
    'actual_sales': y_test.values,
    'predicted_sales': test_predictions,
    'error': residuals,
    'error_percentage': (residuals / y_test.values * 100)
})
results_df.to_csv('linear_regression_predictions.csv', index=False)
print("✅ Predictions saved: linear_regression_predictions.csv")

# Example: Predict next week
# Ensure prediction input matches training feature schema exactly
# This guarantees consistency between training and inference phases
print("\n🔮 EXAMPLE: PREDICTING NEXT WEEK")
print("-" * 40)

# Create next week's date
next_week_date = df['date'].max() + pd.Timedelta(days=7)

# Build feature set EXACTLY like training data
next_week_df = pd.DataFrame({
    'promotion': [1],  # Planning a promotion
    'holiday': [0],    # Not a holiday
    'week_of_year': [next_week_date.isocalendar().week],
    'month': [next_week_date.month],
    'quarter': [(next_week_date.month - 1) // 3 + 1],
    'year': [next_week_date.year],
    'days_since_start': [(next_week_date - df['date'].min()).days],
    'day_of_week': [next_week_date.dayofweek]
})

# Ensure feature order matches training
next_week_df = next_week_df[features]

# Scale features
next_week_scaled = scaler.transform(next_week_df)

# Predict
next_week_prediction = model.predict(next_week_scaled)[0]

print(f"Predicted sales for next week: ${next_week_prediction:,.2f}")

print("\nInput features:")
for col in next_week_df.columns:
    print(f"  • {col}: {next_week_df[col].values[0]}")

print("\n" + "=" * 60)
print("✅ LINEAR REGRESSION MODEL COMPLETE!")
print("=" * 60)
print(f"\nModel Performance Summary:")
print(f"  • Test MAE: ${test_mae:,.2f}")
print(f"  • Test R²: {test_r2:.4f}")
print(f"\nNext step: Run 3_time_series.py for time-series model")
