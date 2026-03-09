import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate 2 years of weekly data
start_date = datetime(2022, 1, 1)
weeks = 104  # 2 years of weekly data

dates = [start_date + timedelta(weeks=i) for i in range(weeks)]

# Create base sales with trend and seasonality
base_sales = 10000  # Base weekly sales
trend = np.linspace(0, 3000, weeks)  # Upward trend
seasonality = 2000 * np.sin(np.linspace(0, 4*np.pi, weeks))  # Seasonal pattern

# Add randomness
noise = np.random.normal(0, 1000, weeks)

# Calculate sales
sales = base_sales + trend + seasonality + noise

# Add features
promotions = np.random.binomial(1, 0.2, weeks)  # 20% chance of promotion
holidays = np.random.binomial(1, 0.05, weeks)   # 5% chance of holiday

# Boost sales during promotions and holidays
sales = sales + (promotions * 2000) + (holidays * 3000)
sales = np.maximum(sales, 5000)  # Minimum sales of 5000

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'sales': sales.astype(int),
    'promotion': promotions,
    'holiday': holidays,
    'temperature': np.random.uniform(10, 35, weeks).round(1),
    'day_of_week': [d.weekday() for d in dates]
})

# Save to CSV
df.to_csv('sales_data.csv', index=False)

print("✅ Sample data created: sales_data.csv")
print(f"\n📊 Data shape: {df.shape}")
print(f"\n📅 Date range: {df['date'].min()} to {df['date'].max()}")
print(f"\n💰 Sales range: ${df['sales'].min():,.0f} to ${df['sales'].max():,.0f}")
print("\nFirst few rows:")
print(df.head(10))
