import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 60)
print("STEP 1: DATA EXPLORATION")
print("=" * 60)

# Load the data
print("\n📂 Loading data...")
df = pd.read_csv('sales_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Basic information
print("\n1️⃣ BASIC INFORMATION")
print("-" * 40)
print(f"Total records: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Columns: {list(df.columns)}")
print(f"\nData types:\n{df.dtypes}")

# Statistical summary
print("\n2️⃣ STATISTICAL SUMMARY")
print("-" * 40)
print(df.describe())

# Check for missing values
print("\n3️⃣ MISSING VALUES")
print("-" * 40)
print(df.isnull().sum())

# Sales statistics
print("\n4️⃣ SALES ANALYSIS")
print("-" * 40)
print(f"Mean weekly sales: ${df['sales'].mean():,.2f}")
print(f"Median weekly sales: ${df['sales'].median():,.2f}")
print(f"Std deviation: ${df['sales'].std():,.2f}")
print(f"Min sales: ${df['sales'].min():,.2f}")
print(f"Max sales: ${df['sales'].max():,.2f}")

# Feature correlation
print("\n5️⃣ CORRELATION WITH SALES")
print("-" * 40)
correlations = df.corr()['sales'].sort_values(ascending=False)
print(correlations)

# Visualization
print("\n📊 Creating visualizations...")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Sales over time
axes[0, 0].plot(df['date'], df['sales'], linewidth=1.5)
axes[0, 0].set_title('Weekly Sales Over Time', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales ($)')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(True, alpha=0.3)

# 2. Sales distribution
axes[0, 1].hist(df['sales'], bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('Sales Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Sales ($)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].axvline(df['sales'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${df["sales"].mean():,.0f}')
axes[0, 1].legend()

# 3. Promotion impact
promo_sales = df.groupby('promotion')['sales'].mean()
axes[1, 0].bar(['No Promotion', 'Promotion'], promo_sales.values, 
               color=['lightblue', 'orange'], edgecolor='black')
axes[1, 0].set_title('Average Sales: Promotion vs No Promotion', 
                     fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Average Sales ($)')
for i, v in enumerate(promo_sales.values):
    axes[1, 0].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

# 4. Holiday impact
holiday_sales = df.groupby('holiday')['sales'].mean()
axes[1, 1].bar(['Regular Day', 'Holiday'], holiday_sales.values, 
               color=['lightgreen', 'red'], edgecolor='black')
axes[1, 1].set_title('Average Sales: Holiday vs Regular Day', 
                     fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Average Sales ($)')
for i, v in enumerate(holiday_sales.values):
    axes[1, 1].text(i, v + 500, f'${v:,.0f}', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('1_data_exploration.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: 1_data_exploration.png")

# Additional time-based analysis
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 5))

# Monthly average
# Extract day of week from date to analyze weekly sales patterns
# (Helps identify which days drive the most revenue)
df['month'] = df['date'].dt.month
# Create day of week feature (0=Monday, 6=Sunday)
df['day_of_week'] = df['date'].dt.dayofweek
monthly_avg = df.groupby('month')['sales'].mean()
axes2[0].plot(monthly_avg.index, monthly_avg.values, marker='o', linewidth=2)
axes2[0].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
axes2[0].set_xlabel('Month')
axes2[0].set_ylabel('Average Sales ($)')
axes2[0].set_xticks(range(1, 13))
axes2[0].grid(True, alpha=0.3)

# Day of week pattern
dow_avg = df.groupby('day_of_week')['sales'].mean()
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
axes2[1].bar(range(7), dow_avg.values, color='skyblue', edgecolor='black')
axes2[1].set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
axes2[1].set_xlabel('Day of Week')
axes2[1].set_ylabel('Average Sales ($)')
axes2[1].set_xticks(range(7))
axes2[1].set_xticklabels(days)

plt.tight_layout()
plt.savefig('1_time_patterns.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: 1_time_patterns.png")

print("\n" + "=" * 60)
print("✅ EXPLORATION COMPLETE!")
print("=" * 60)
print("\nKey Insights:")
print(f"• Dataset spans {len(df)} weeks")
print(f"• Average weekly sales: ${df['sales'].mean():,.2f}")
print(f"• Sales variation (std dev): ${df['sales'].std():,.2f}")
print(f"• Promotions boost sales by: ${promo_sales.iloc[1] - promo_sales.iloc[0]:,.2f}")
print(f"• Holidays boost sales by: ${holiday_sales.iloc[1] - holiday_sales.iloc[0]:,.2f}")
print("\nNext step: Run 2_linear_regression.py")
