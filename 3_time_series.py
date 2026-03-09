import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 3: TIME-SERIES FORECASTING WITH PROPHET")
print("=" * 60)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Prepare data for Prophet
# Prophet requires columns named 'ds' (date) and 'y' (target)
print("\n1️⃣ PREPARING DATA FOR PROPHET")
print("-" * 40)
prophet_df = df[['date', 'sales']].rename(columns={'date': 'ds', 'sales': 'y'})
print(f"Data shape: {prophet_df.shape}")
print("✅ Data formatted with 'ds' (date) and 'y' (sales) columns")

# Split into train/test
split_idx = int(len(prophet_df) * 0.8)
train_df = prophet_df[:split_idx]
test_df = prophet_df[split_idx:]

print(f"\nTraining set: {len(train_df)} weeks")
print(f"Test set: {len(test_df)} weeks")
print(f"Training period: {train_df['ds'].min()} to {train_df['ds'].max()}")
print(f"Test period: {test_df['ds'].min()} to {test_df['ds'].max()}")

# Create and train Prophet model
print("\n2️⃣ CREATING PROPHET MODEL")
print("-" * 40)
print("Prophet will automatically detect:")
print("  • Yearly seasonality")
print("  • Weekly seasonality")
print("  • Trends in the data")

model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive',  # Can be 'additive' or 'multiplicative'
    changepoint_prior_scale=0.05  # Flexibility of trend changes (default: 0.05)
)

# Add custom features (regressors)
print("\n3️⃣ ADDING ADDITIONAL FEATURES")
print("-" * 40)

# Prepare full dataframe with features
full_df = df[['date', 'sales', 'promotion', 'holiday']].rename(
    columns={'date': 'ds', 'sales': 'y'}
)

# Add regressors to the model
model.add_regressor('promotion')
model.add_regressor('holiday')
print("✅ Added 'promotion' and 'holiday' as additional predictors")

# Fit the model
print("\n4️⃣ TRAINING THE MODEL")
print("-" * 40)
train_full = full_df[:split_idx]
test_full = full_df[split_idx:]

print("Training model... (this may take a minute)")
model.fit(train_full)
print("✅ Model trained successfully!")

# Make predictions on test set
print("\n5️⃣ MAKING PREDICTIONS")
print("-" * 40)
# Create future dataframe for test period
future = test_full[['ds', 'promotion', 'holiday']]
forecast = model.predict(future)

# Extract predictions
predictions = forecast['yhat'].values
actuals = test_full['y'].values

print(f"✅ Generated {len(predictions)} predictions")

# Evaluate the model
print("\n6️⃣ MODEL PERFORMANCE")
print("-" * 40)

mae = mean_absolute_error(actuals, predictions)
rmse = np.sqrt(mean_squared_error(actuals, predictions))
r2 = r2_score(actuals, predictions)

print(f"  • MAE (Mean Absolute Error): ${mae:,.2f}")
print(f"  • RMSE (Root Mean Squared Error): ${rmse:,.2f}")
print(f"  • R² Score: {r2:.4f}")

print("\n💡 Interpretation:")
print(f"  • On average, predictions are off by ${mae:,.2f}")
print(f"  • Model explains {r2*100:.1f}% of sales variation")

# Forecast future weeks
print("\n7️⃣ FORECASTING FUTURE WEEKS")
print("-" * 40)

# Create future dates (next 12 weeks)
last_date = full_df['ds'].max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), 
                              periods=12, freq='W')

# Assume no promotions/holidays for future (you can change this)
future_forecast = pd.DataFrame({
    'ds': future_dates,
    'promotion': 0,
    'holiday': 0
})

future_predictions = model.predict(future_forecast)
print(f"✅ Forecasted next {len(future_dates)} weeks")

# Visualizations
print("\n8️⃣ CREATING VISUALIZATIONS")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Actual vs Predicted on Test Set
axes[0, 0].plot(test_full['ds'], actuals, label='Actual', marker='o', linewidth=2)
axes[0, 0].plot(test_full['ds'], predictions, label='Predicted', 
                marker='s', linewidth=2, alpha=0.7)
axes[0, 0].fill_between(test_full['ds'], 
                         forecast['yhat_lower'].values,
                         forecast['yhat_upper'].values,
                         alpha=0.2, label='Confidence Interval')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].set_title('Prophet: Test Set Predictions', fontweight='bold', fontsize=14)
axes[0, 0].legend()
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. Forecast for future weeks
axes[0, 1].plot(full_df['ds'].tail(30), full_df['y'].tail(30), 
                label='Historical', marker='o', linewidth=2)
axes[0, 1].plot(future_predictions['ds'], future_predictions['yhat'],
                label='Forecast', marker='s', linewidth=2, color='red')
axes[0, 1].fill_between(future_predictions['ds'],
                         future_predictions['yhat_lower'],
                         future_predictions['yhat_upper'],
                         alpha=0.2, color='red')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Sales ($)')
axes[0, 1].set_title('12-Week Sales Forecast', fontweight='bold', fontsize=14)
axes[0, 1].legend()
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(True, alpha=0.3)

# 3. Residuals
residuals = actuals - predictions
axes[1, 0].scatter(predictions, residuals, alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Predicted Sales ($)')
axes[1, 0].set_ylabel('Residuals ($)')
axes[1, 0].set_title('Residual Plot', fontweight='bold', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 4. Error distribution
axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 1].set_xlabel('Error ($)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Prediction Error Distribution', fontweight='bold', fontsize=14)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('3_prophet_results.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: 3_prophet_results.png")

# Component plots (trend, seasonality)
print("\nCreating component plots...")
fig2 = model.plot_components(future_predictions)
plt.savefig('3_prophet_components.png', dpi=300, bbox_inches='tight')
print("✅ Component plot saved: 3_prophet_components.png")

# Save predictions
print("\n9️⃣ SAVING RESULTS")
print("-" * 40)

# Test predictions
test_results = pd.DataFrame({
    'date': test_full['ds'],
    'actual_sales': actuals,
    'predicted_sales': predictions,
    'lower_bound': forecast['yhat_lower'].values,
    'upper_bound': forecast['yhat_upper'].values,
    'error': residuals
})
test_results.to_csv('prophet_test_predictions.csv', index=False)
print("✅ Test predictions saved: prophet_test_predictions.csv")

# Future forecast
future_results = pd.DataFrame({
    'date': future_predictions['ds'],
    'predicted_sales': future_predictions['yhat'],
    'lower_bound': future_predictions['yhat_lower'],
    'upper_bound': future_predictions['yhat_upper']
})
future_results.to_csv('prophet_future_forecast.csv', index=False)
print("✅ Future forecast saved: prophet_future_forecast.csv")

# Print future forecast
print("\n🔮 NEXT 12 WEEKS FORECAST:")
print("-" * 40)
for idx, row in future_results.iterrows():
    print(f"Week {idx+1} ({row['date'].strftime('%Y-%m-%d')}): "
          f"${row['predicted_sales']:,.0f} "
          f"(${row['lower_bound']:,.0f} - ${row['upper_bound']:,.0f})")

print("\n" + "=" * 60)
print("✅ TIME-SERIES MODEL COMPLETE!")
print("=" * 60)
print(f"\nModel Performance Summary:")
print(f"  • Test MAE: ${mae:,.2f}")
print(f"  • Test R²: {r2:.4f}")
print(f"\nNext step: Run 4_model_comparison.py to compare both models")
