Sales Prediction System - Complete Guide
A step-by-step system to predict weekly sales using Linear Regression and Time-Series (Prophet) models.
What You Need:
Python 3.11 or 3.12** (recommended)

Step 4: Install Required Packages
pip install...

pandas==2.1.4
numpy==1.26.2
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
prophet==1.1.5
jupyter==1.0.0

Option 1: Run Everything at Once

python main.py

This runs the complete pipeline automatically.

Option 2: Run Step by Step
```bash
# Step 0: Generate sample data
python generate_sample_data.py

# Step 1: Explore the data
python 1_explore_data.py

# Step 2: Build Linear Regression model
python 2_linear_regression.py

# Step 3: Build Prophet time-series model
python 3_time_series.py

# Step 4: Compare both models
python 4_model_comparison.py
```

Expected Output:
After running all steps, you'll have:
6 visualization images (PNG files)
4 CSV files with predictions
1 comparison report (TXT file)

Detailed Walkthrough

Script 1: `generate_sample_data.py`

**What it does:**
- Creates 2 years of weekly sales data
- Adds realistic patterns (trends, seasonality, randomness)
- Includes features: promotions, holidays, temperature

**Output:** `sales_data.csv`

**Data format:**
```csv
date,sales,promotion,holiday,temperature,day_of_week
2022-01-01,15234,0,1,22.3,5
2022-01-08,12456,1,0,18.7,6
...
```

---

Script 2: `1_explore_data.py`

**What it does:**
- Loads and analyzes the data
- Calculates statistics (mean, median, std dev)
- Creates 6 visualizations showing:
  - Sales over time
  - Sales distribution
  - Promotion impact
  - Holiday impact
  - Monthly patterns
  - Day-of-week patterns

**Key Outputs:**
- `1_data_exploration.png` - 4 charts overview
- `1_time_patterns.png` - Time-based patterns

Script 3: `2_linear_regression.py`

**What it does:**
- Creates features from the date (month, week, etc.)
- Splits data into training (80%) and test (20%) sets
- Trains a Linear Regression model
- Makes predictions and evaluates performance

**Key Concepts:**

**Features used:**
- `promotion`: Binary (0 or 1)
- `holiday`: Binary (0 or 1)
- `temperature`: Continuous
- `day_of_week`: 0-6 (Monday-Sunday)
- `week_of_year`: 1-52
- `month`: 1-12
- `quarter`: 1-4
- `days_since_start`: Captures trend

**Performance Metrics:**
- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **R²** (R-squared): How much variation the model explains (0-1, higher is better)

**Outputs:**
- `2_linear_regression_results.png` - 4 charts showing predictions
- `linear_regression_predictions.csv` - Detailed predictions

**Example interpretation:**
```
MAE: $1,234 → On average, predictions are off by $1,234
R²: 0.85 → Model explains 85% of sales variation
```

---
Script 4: `3_time_series.py`

**What it does:**
- Uses Facebook's Prophet algorithm
- Automatically detects trends and seasonality
- Provides confidence intervals
- Forecasts future weeks

**Prophet Advantages:**
- Handles missing data
- Automatically detects yearly/weekly patterns
- Provides uncertainty estimates
- Great for data with strong time patterns

**Outputs:**
- `3_prophet_results.png` - Predictions with confidence intervals
- `3_prophet_components.png` - Trend and seasonality breakdown
- `prophet_test_predictions.csv` - Test set predictions
- `prophet_future_forecast.csv` - Next 12 weeks forecast

**Understanding Prophet Components:**
- **Trend**: Overall direction (up/down/flat)
- **Yearly seasonality**: Pattern that repeats annually
- **Weekly seasonality**: Pattern within each week

Script 5: `4_model_comparison.py`

**What it does:**
- Compares Linear Regression vs Prophet
- Creates side-by-side visualizations
- Generates detailed report
- Recommends which model to use

**Outputs:**
- `4_model_comparison.png` - Visual comparison
- `model_comparison_report.txt` - Detailed analysis

**Decision Guide:**

Use **Linear Regression** if:
You have many relevant features
Relationships are relatively stable
You need fast predictions
Interpretability is important

Use **Prophet** if:
Time patterns dominate
You need confidence intervals
Limited additional features
Strong seasonality exists

Using Your Own imaginary Data

Step 1: Prepare Your CSV

Your file must have **at minimum**:
- `date` column (format: YYYY-MM-DD)
- `sales` column (numeric)

**Optional but recommended:**
- `promotion` (0 or 1)
- `holiday` (0 or 1)
- Any other relevant features

Example:
```csv
date,sales,promotion,holiday
2022-01-01,15000,0,1
2022-01-08,12000,1,0
2022-01-15,13500,0,0
```

Step 2: Modify the Scripts

In each script, change:
```python
df = pd.read_csv('sales_data.csv')
```
to:
```python
df = pd.read_csv('YOUR_FILE.csv')
```

Step 3: Adjust Feature Names

If your columns have different names:
```python
# Original
df = df[['date', 'sales', 'promotion', 'holiday']]

# Your version
df = df[['order_date', 'revenue', 'promo_flag', 'is_holiday']]
df = df.rename(columns={
    'order_date': 'date',
    'revenue': 'sales',
    'promo_flag': 'promotion',
    'is_holiday': 'holiday'
})

Understanding the Models

Linear Regression

How it works:**

sales = β₀ + β₁×promotion + β₂×holiday + β₃×temperature + ...

**Pros:**
- Fast training and prediction
- Interpretable coefficients
- Works with many features
- Good baseline model

**Cons:**
- Assumes linear relationships
- May miss complex patterns
- Sensitive to outliers

**When predictions fail:**
- Features don't capture patterns
- Non-linear relationships exist
- Data has changed significantly

Prophet (Time-Series)

**How it works:**

y(t) = g(t) + s(t) + h(t) + ε(t)

g(t) = trend
s(t) = seasonality
h(t) = holidays
ε(t) = error


**Pros:**
- Handles seasonality automatically
- Robust to missing data
- Provides uncertainty intervals
- Great for forecasting

**Cons:**
- Requires time-ordered data
- Less flexible with features
- Slower training
- Assumes patterns continue

**When predictions fail:**
- Patterns change suddenly
- External shocks occur
- Limited historical data


```

Expected Results

Sample Performance (on generated data):

**Linear Regression:**
- MAE: ~$1,200-1,500
- R²: ~0.75-0.85
- Training time: < 1 second

**Prophet:**
- MAE: ~$1,000-1,400
- R²: ~0.80-0.90
- Training time: 10-30 seconds

**Note:** Actual results vary based on the data!


1. **Experiment with features:**
   - Add more predictors (competitor sales, weather, events)
   - Try polynomial features
   - Test different feature combinations

2. **Try other models:**
   - Random Forest
   - XGBoost
   - LSTM (deep learning)

3. **Improve predictions:**
   - Collect more data
   - Add domain knowledge
   - Handle outliers
   - Cross-validation

4. **Deploy the model:**
   - Save trained model
   - Create API endpoint
   - Build dashboard
   - Automate weekly predictions
