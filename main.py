import os
import subprocess
import sys

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print("\n" + "=" * 70)
    print(f"RUNNING: {script_name}")
    print(f"Description: {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=False, 
                              text=True, 
                              check=True)
        print(f"✅ {script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {script_name}")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"❌ {script_name} not found!")
        return False

def main():
    print("=" * 70)
    print("SALES PREDICTION SYSTEM - FULL PIPELINE")
    print("=" * 70)
    print("\nThis script will run the entire pipeline:")
    print("1. Generate sample sales data")
    print("2. Explore and visualize the data")
    print("3. Build Linear Regression model")
    print("4. Build Prophet time-series model")
    print("5. Compare both models")
    
    response = input("\nDo you want to proceed? (yes/no): ").lower()
    
    if response not in ['yes', 'y']:
        print("Pipeline cancelled.")
        return
    
    # Check if data file exists
    if os.path.exists('sales_data.csv'):
        response = input("\nsales_data.csv already exists. Regenerate? (yes/no): ").lower()
        if response in ['yes', 'y']:
            run_data = True
        else:
            run_data = False
            print("Using existing sales_data.csv")
    else:
        run_data = True
    
    # Pipeline steps
    steps = [
        ('generate_sample_data.py', 'Generate sample sales data', run_data),
        ('1_explore_data.py', 'Explore and visualize data', True),
        ('2_linear_regression.py', 'Build Linear Regression model', True),
        ('3_time_series.py', 'Build Prophet time-series model', True),
        ('4_model_comparison.py', 'Compare both models', True),
    ]
    
    completed = []
    failed = []
    
    for script, description, should_run in steps:
        if should_run:
            success = run_script(script, description)
            if success:
                completed.append(script)
            else:
                failed.append(script)
                print(f"\n⚠️ Warning: {script} failed. Continuing to next step...")
        else:
            print(f"\n⏭️ Skipping {script}")
    
    # Summary
    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(f"\n✅ Completed: {len(completed)} steps")
    for script in completed:
        print(f"   • {script}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} steps")
        for script in failed:
            print(f"   • {script}")
    else:
        print("\n🎉 All steps completed successfully!")
    
    print("\n📁 Generated files:")
    files = [
        'sales_data.csv',
        '1_data_exploration.png',
        '1_time_patterns.png',
        '2_linear_regression_results.png',
        'linear_regression_predictions.csv',
        '3_prophet_results.png',
        '3_prophet_components.png',
        'prophet_test_predictions.csv',
        'prophet_future_forecast.csv',
        '4_model_comparison.png',
        'model_comparison_report.txt'
    ]
    
    for file in files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"   ✅ {file} ({size:,} bytes)")
        else:
            print(f"   ❌ {file} (not found)")
    
    print("\n" + "=" * 70)
    print("Next steps:")
    print("1. Check the visualizations (PNG files)")
    print("2. Read model_comparison_report.txt for detailed analysis")
    print("3. Use the CSV files to review predictions")
    print("4. Modify the scripts to use your own data!")
    print("=" * 70)

if __name__ == "__main__":
    main()
