import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🔮 SALES PREDICTION TOOL")
print("=" * 70)

# Load the historical data
print("\n📂 Loading historical data...")
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Prepare data for Prophet
prophet_df = df[['date', 'sales', 'promotion', 'holiday']].rename(
    columns={'date': 'ds', 'sales': 'y'}
)

# Train Prophet model on ALL data (not just training set)
print("🤖 Training Prophet model on all historical data...")
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='additive'
)

model.add_regressor('promotion')
model.add_regressor('holiday')
model.fit(prophet_df)
print("✅ Model trained successfully!\n")

# Interactive prediction function
def predict_sales():
    print("=" * 70)
    print("ENTER PREDICTION DETAILS")
    print("=" * 70)
    
    # Get date
    print("\n📅 Enter the date you want to predict:")
    print("   Format: YYYY-MM-DD (example: 2024-03-15)")
    date_input = input("   Date: ").strip()
    
    try:
        predict_date = pd.to_datetime(date_input)
    except:
        print("❌ Invalid date format! Please use YYYY-MM-DD")
        return
    
    # Get promotion status
    print("\n🎯 Will there be a promotion that week?")
    promo_input = input("   Enter 1 for Yes, 0 for No: ").strip()
    promotion = int(promo_input) if promo_input in ['0', '1'] else 0
    
    # Get holiday status
    print("\n🎊 Will it be a holiday week?")
    holiday_input = input("   Enter 1 for Yes, 0 for No: ").strip()
    holiday = int(holiday_input) if holiday_input in ['0', '1'] else 0
    
    # Create prediction dataframe
    future = pd.DataFrame({
        'ds': [predict_date],
        'promotion': [promotion],
        'holiday': [holiday]
    })
    
    # Make prediction
    print("\n🔮 Calculating prediction...")
    forecast = model.predict(future)
    
    predicted_sales = forecast['yhat'].values[0]
    lower_bound = forecast['yhat_lower'].values[0]
    upper_bound = forecast['yhat_upper'].values[0]
    
    # Display results
    print("\n" + "=" * 70)
    print("📊 PREDICTION RESULTS")
    print("=" * 70)
    print(f"\n📅 Date: {predict_date.strftime('%A, %B %d, %Y')}")
    print(f"🎯 Promotion: {'Yes' if promotion else 'No'}")
    print(f"🎊 Holiday: {'Yes' if holiday else 'No'}")
    print(f"\n💰 PREDICTED SALES: ${predicted_sales:,.2f}")
    print(f"\n📈 Confidence Interval:")
    print(f"   • Lower bound: ${lower_bound:,.2f}")
    print(f"   • Upper bound: ${upper_bound:,.2f}")
    
    # Calculate impact
    base_forecast = model.predict(pd.DataFrame({
        'ds': [predict_date],
        'promotion': [0],
        'holiday': [0]
    }))
    base_sales = base_forecast['yhat'].values[0]
    
    if promotion:
        promo_impact = predicted_sales - model.predict(pd.DataFrame({
            'ds': [predict_date],
            'promotion': [0],
            'holiday': [holiday]
        }))['yhat'].values[0]
        print(f"\n🎯 Promotion impact: +${promo_impact:,.2f}")
    
    if holiday:
        holiday_impact = predicted_sales - model.predict(pd.DataFrame({
            'ds': [predict_date],
            'promotion': [promotion],
            'holiday': [0]
        }))['yhat'].values[0]
        print(f"🎊 Holiday impact: +${holiday_impact:,.2f}")
    
    print("\n" + "=" * 70)
    
    # Save prediction
    save = input("\n💾 Save this prediction to CSV? (y/n): ").lower()
    if save == 'y':
        result = pd.DataFrame({
            'date': [predict_date],
            'predicted_sales': [predicted_sales],
            'lower_bound': [lower_bound],
            'upper_bound': [upper_bound],
            'promotion': [promotion],
            'holiday': [holiday]
        })
        
        # Append to file or create new
        try:
            existing = pd.read_csv('my_predictions.csv')
            result = pd.concat([existing, result], ignore_index=True)
        except FileNotFoundError:
            pass
        
        result.to_csv('my_predictions.csv', index=False)
        print("✅ Saved to my_predictions.csv")

# Main loop
def main():
    print("\n💡 Historical data summary:")
    print(f"   • Data range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"   • Average sales: ${df['sales'].mean():,.2f}")
    print(f"   • Sales range: ${df['sales'].min():,.0f} - ${df['sales'].max():,.0f}")
    
    while True:
        predict_sales()
        
        print("\n" + "=" * 70)
        another = input("\n🔄 Make another prediction? (y/n): ").lower()
        if another != 'y':
            print("\n👋 Thank you for using the Sales Prediction Tool!")
            print("=" * 70)
            break

if __name__ == "__main__":
    main()
