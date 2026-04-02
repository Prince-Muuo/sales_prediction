# 🚀 Local Installation Guide - Sales Prediction Dashboard

Complete step-by-step guide to clone and run this project on your local machine.

---

## 📋 Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Running the Dashboard](#running-the-dashboard)
4. [Running Individual Scripts](#running-individual-scripts)
5. [Troubleshooting](#troubleshooting)
6. [Project Structure](#project-structure)

---

## ✅ Prerequisites

Before you begin, ensure you have the following installed:

### Required Software:
- **Python 3.11 or higher** (3.13 recommended)
- **Git** (for cloning the repository)
- **pip** (comes with Python)

### Check Your Installations:

**Windows Command Prompt / Mac-Linux Terminal:**
```bash
python --version
# Should show: Python 3.11.x or higher

pip --version
# Should show: pip 23.x.x or higher

git --version
# Should show: git version 2.x.x
```

---

## 📥 Installation Steps

### STEP 1: Clone the Repository

**Open Command Prompt (Windows) or Terminal (Mac/Linux)**

```bash
# Navigate to where you want to store the project
cd Desktop

# Clone the repository
git clone https://github.com/Prince-Muuo/sales_prediction.git

# Navigate into the project folder
cd sales_prediction
```

**Verify you're in the right folder:**
```bash
# Windows
dir

# Mac/Linux
ls
```

You should see files like: `dashboard.py`, `sales_data.csv`, `requirements.txt`, etc.

---

### STEP 2: Create Virtual Environment

**Why?** Keeps project dependencies isolated from your system Python.

**Create virtual environment:**

```bash
# Windows
python -m venv venv

# Mac/Linux
python3 -m venv venv
```

**Wait 30-60 seconds for it to complete.**

---

### STEP 3: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

**You'll know it worked when you see `(venv)` at the beginning of your command line:**
```
(venv) C:\Users\YourName\Desktop\sales_prediction>
```

---

### STEP 4: Install Dependencies

**Install all required packages:**

```bash
pip install -r requirements.txt
```

**This will install:**
- streamlit (web framework)
- pandas (data manipulation)
- numpy (numerical computing)
- scikit-learn (machine learning)
- matplotlib (visualization)
- seaborn (statistical visualization)
- prophet (time-series forecasting)
- plotly (interactive charts)
- cmdstanpy (Prophet dependency)

**Installation time:** 5-10 minutes depending on internet speed.

**You'll see lots of text scrolling - this is normal!**

---

### STEP 5: Verify Installation

**Check that everything installed correctly:**

```bash
pip list
```

You should see all the packages listed above.

**Quick test:**
```bash
python -c "import streamlit, pandas, prophet, plotly; print('All packages imported successfully!')"
```

If you see `All packages imported successfully!` - you're good to go! ✅

---

## 🎯 Running the Dashboard

### Method 1: Web Dashboard (Recommended)

**Run the Streamlit web application:**

```bash
streamlit run dashboard.py
```

**What happens:**
- A browser window automatically opens
- Dashboard loads at `http://localhost:8501`
- You can interact with the web interface

**To stop the dashboard:**
- Press `Ctrl+C` in the terminal

---

### Method 2: Interactive Prediction Tool

**Run the command-line prediction tool:**

```bash
python predict_future.py
```

**What it does:**
- Asks for date, promotion, holiday status
- Returns predicted sales
- Saves predictions to CSV

**Example interaction:**
```
📅 Enter the date you want to predict:
   Format: YYYY-MM-DD (example: 2024-03-15)
   Date: 2024-06-15
   
🎯 Will there be a promotion that week?
   Enter 1 for Yes, 0 for No: 1
   
🎊 Will it be a holiday week?
   Enter 1 for Yes, 0 for No: 0
   
💰 PREDICTED SALES: $15,234.56
```

---

## 📊 Running Individual Scripts

### Data Exploration
```bash
python 1_explore_data.py
```
**Output:** 
- `1_data_exploration.png`
- `1_time_patterns.png`
- Statistical analysis in terminal

---

### Linear Regression Model
```bash
python 2_linear_regression.py
```
**Output:**
- `2_linear_regression_results.png`
- `linear_regression_predictions.csv`
- Model performance metrics

---

### Prophet Time-Series Model
```bash
python 3_time_series.py
```
**Output:**
- `3_prophet_results.png`
- `3_prophet_components.png`
- `prophet_test_predictions.csv`
- `prophet_future_forecast.csv`

---

### Model Comparison
```bash
python 4_model_comparison.py
```
**Output:**
- `4_model_comparison.png`
- `model_comparison_report.txt`
- Performance comparison

---

### Generate Sample Data (Optional)
```bash
python generate_sample_data.py
```
**Creates:** Fresh `sales_data.csv` with 2 years of synthetic data

---

### Visualize Your Predictions
```bash
# First make some predictions with predict_future.py
# Then run:
python visualize_predictions.py
```
**Output:**
- `prediction_visualization.png`
- `monthly_comparison.png`

---

## 🔄 Complete Pipeline

**Run everything in sequence:**

```bash
# Generate fresh data (optional)
python generate_sample_data.py

# Explore the data
python 1_explore_data.py

# Train Linear Regression
python 2_linear_regression.py

# Train Prophet model
python 3_time_series.py

# Compare models
python 4_model_comparison.py

# Make predictions
python predict_future.py

# Visualize predictions
python visualize_predictions.py

# Run web dashboard
streamlit run dashboard.py
```

**OR use the automated pipeline:**

```bash
python main.py
```

This runs all scripts automatically!

---

## 🗂️ Project Structure

```
sales_prediction/
│
├── 📊 Data Files
│   ├── sales_data.csv                    # Historical sales data
│   ├── my_predictions.csv                # Your saved predictions
│   └── *.csv                             # Other generated data files
│
├── 🐍 Python Scripts
│   ├── dashboard.py                      # Streamlit web dashboard
│   ├── generate_sample_data.py           # Create sample data
│   ├── 1_explore_data.py                 # Data exploration
│   ├── 2_linear_regression.py            # Linear model
│   ├── 3_time_series.py                  # Prophet model
│   ├── 4_model_comparison.py             # Model comparison
│   ├── predict_future.py                 # Interactive predictions
│   ├── visualize_predictions.py          # Prediction visualizer
│   └── main.py                           # Run all scripts
│
├── 📈 Visualizations (Generated)
│   ├── 1_data_exploration.png
│   ├── 1_time_patterns.png
│   ├── 2_linear_regression_results.png
│   ├── 3_prophet_results.png
│   ├── 3_prophet_components.png
│   ├── 4_model_comparison.png
│   ├── prediction_visualization.png
│   └── monthly_comparison.png
│
├── 📁 Configuration
│   ├── requirements.txt                  # Python dependencies
│   ├── .gitignore                        # Git ignore rules
│   ├── README.md                         # Project documentation
│   └── .streamlit/
│       └── config.toml                   # Streamlit configuration
│
└── 📂 Environment
    └── venv/                             # Virtual environment (created locally)
```

---

## 🐛 Troubleshooting

### Issue 1: "Python not found"

**Windows:**
```bash
# Use full path
C:\Users\YourName\AppData\Local\Programs\Python\Python313\python.exe -m venv venv
```

**Mac/Linux:**
```bash
# Try python3 instead
python3 -m venv venv
```

---

### Issue 2: "Module not found" errors

**Solution:**
```bash
# Make sure virtual environment is activated (you see (venv))
# Then reinstall packages
pip install -r requirements.txt
```

---

### Issue 3: Prophet installation fails

**Windows:**
```bash
# Install Visual C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Then:
pip install prophet
```

**Mac:**
```bash
# Install gcc
brew install gcc
pip install prophet
```

**Linux:**
```bash
sudo apt-get install build-essential
pip install prophet
```

---

### Issue 4: Streamlit won't start

**Check port availability:**
```bash
streamlit run dashboard.py --server.port 8502
```

**Clear Streamlit cache:**
```bash
streamlit cache clear
```

---

### Issue 5: "sales_data.csv not found"

**Make sure you're in the correct directory:**
```bash
# Check current directory
cd  # Windows
pwd # Mac/Linux

# Navigate to project folder
cd path/to/sales_prediction
```

**Generate sample data:**
```bash
python generate_sample_data.py
```

---

### Issue 6: Virtual environment activation fails

**Windows PowerShell restriction:**
```powershell
# Run as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Then activate normally
venv\Scripts\activate
```

---

### Issue 7: Permission denied errors

**Windows:**
```bash
# Run Command Prompt as Administrator
```

**Mac/Linux:**
```bash
# Use sudo for pip
sudo pip install -r requirements.txt

# Or use --user flag
pip install --user -r requirements.txt
```

---

## 🔧 Advanced Configuration

### Change Streamlit Port
```bash
streamlit run dashboard.py --server.port 8080
```

### Run in Headless Mode (No browser)
```bash
streamlit run dashboard.py --server.headless true
```

### Enable Wide Mode by Default
Edit `.streamlit/config.toml`:
```toml
[theme]
base="light"

[server]
port=8501
```

---

## 🎨 Customization

### Use Your Own Data

**Replace `sales_data.csv` with your data.**

**Required format:**
```csv
date,sales,promotion,holiday,temperature,day_of_week
2022-01-01,15000,0,1,22.3,5
2022-01-08,12000,1,0,18.7,6
```

**Minimum columns:**
- `date` (YYYY-MM-DD format)
- `sales` (numeric)

**Optional columns:**
- `promotion` (0 or 1)
- `holiday` (0 or 1)
- `temperature` (numeric)
- `day_of_week` (0-6)

**Then retrain models:**
```bash
python 2_linear_regression.py
python 3_time_series.py
```

---

## 💡 Tips for Success

### 1. Always Activate Virtual Environment
Before running any script:
```bash
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 2. Keep Dependencies Updated
```bash
pip list --outdated
pip install --upgrade package-name
```

### 3. Save Your Work
```bash
git add .
git commit -m "Description of changes"
git push
```

### 4. Check Logs
If something fails, check terminal output for error messages.

### 5. Test Incrementally
Run scripts one at a time to isolate issues.

---

## 📚 Learning Resources

**Streamlit:**
- Documentation: https://docs.streamlit.io
- Gallery: https://streamlit.io/gallery

**Prophet:**
- Documentation: https://facebook.github.io/prophet/
- Quick Start: https://facebook.github.io/prophet/docs/quick_start.html

**Pandas:**
- Documentation: https://pandas.pydata.org/docs/

**Scikit-learn:**
- Documentation: https://scikit-learn.org/stable/

---

## 🎯 Quick Reference Commands

```bash
# Activate environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Deactivate environment
deactivate

# Install packages
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py

# Run prediction tool
python predict_future.py

# Run all scripts
python main.py

# Check installed packages
pip list

# Update a package
pip install --upgrade package-name

# Clear Python cache
find . -type d -name __pycache__ -exec rm -r {} +  # Mac/Linux
# Windows: Delete __pycache__ folders manually
```

---

## ✅ Installation Checklist

Before running the project:

- [ ] Python 3.11+ installed
- [ ] Git installed
- [ ] Repository cloned
- [ ] Virtual environment created
- [ ] Virtual environment activated (see `(venv)`)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] All packages import successfully
- [ ] `sales_data.csv` exists
- [ ] Ready to run!

---

## 🆘 Getting Help

**If you encounter issues:**

1. **Check error message** - Read what it says carefully
2. **Search the error** - Google the error message
3. **Check GitHub Issues** - See if others had the same problem
4. **Ask for help** - Open an issue on GitHub with:
   - Error message
   - What you were trying to do
   - Your Python version
   - Your operating system

---

## 📞 Support

For issues specific to this project:
- Open an issue: https://github.com/Prince-Muuo/sales_prediction/issues

For general Python/Streamlit help:
- Streamlit Forum: https://discuss.streamlit.io
- Stack Overflow: https://stackoverflow.com

---

## 🎉 You're All Set!

Your sales prediction dashboard is ready to run!

**Quick start:**
```bash
cd sales_prediction
venv\Scripts\activate
streamlit run dashboard.py
```

**Enjoy predicting sales!** 📊🚀

---

**Last Updated:** 2024
**Version:** 1.0
**Author:** Prince Muuo
**License:** MIT
