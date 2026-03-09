import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Sales Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    h1 {
        color: #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = None

# Load and train model function
@st.cache_resource
def load_and_train_model():
    """Load data and train Prophet model"""
    try:
        df = pd.read_csv('sales_data.csv')
        df['date'] = pd.to_datetime(df['date'])
        
        # Prepare for Prophet
        prophet_df = df[['date', 'sales', 'promotion', 'holiday']].rename(
            columns={'date': 'ds', 'sales': 'y'}
        )
        
        # Train model
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='additive'
        )
        model.add_regressor('promotion')
        model.add_regressor('holiday')
        model.fit(prophet_df)
        
        return model, df
    except FileNotFoundError:
        return None, None

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/000000/analytics.png", width=100)
    st.title("📊 Sales Predictor")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Single Prediction", "📊 Batch Predictions", 
         "📈 Model Analytics", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("AI-powered sales forecasting system using Facebook Prophet")
    
    # Load model
    if st.button("🔄 Reload Model"):
        st.cache_resource.clear()
        st.rerun()

# Load model and data
model, historical_data = load_and_train_model()

if model is None:
    st.error("⚠️ sales_data.csv not found! Please make sure the file exists in the same directory.")
    st.stop()

st.session_state.model = model
st.session_state.historical_data = historical_data

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.title("📊 Sales Prediction Dashboard")
    st.markdown("### Welcome to your AI-powered sales forecasting system!")
    
    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Data Range",
            value=f"{len(historical_data)} weeks",
            delta="2 years"
        )
    
    with col2:
        avg_sales = historical_data['sales'].mean()
        st.metric(
            label="💰 Avg Weekly Sales",
            value=f"${avg_sales:,.0f}"
        )
    
    with col3:
        max_sales = historical_data['sales'].max()
        st.metric(
            label="📈 Peak Sales",
            value=f"${max_sales:,.0f}"
        )
    
    with col4:
        std_sales = historical_data['sales'].std()
        st.metric(
            label="📊 Volatility",
            value=f"${std_sales:,.0f}"
        )
    
    st.markdown("---")
    
    # Historical sales chart
    st.subheader("📈 Historical Sales Trend")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical_data['date'],
        y=historical_data['sales'],
        mode='lines+markers',
        name='Sales',
        line=dict(color='#1f77b4', width=2),
        marker=dict(size=6)
    ))
    
    # Highlight promotions
    promo_data = historical_data[historical_data['promotion'] == 1]
    fig.add_trace(go.Scatter(
        x=promo_data['date'],
        y=promo_data['sales'],
        mode='markers',
        name='Promotion',
        marker=dict(size=12, color='orange', symbol='star')
    ))
    
    # Highlight holidays
    holiday_data = historical_data[historical_data['holiday'] == 1]
    fig.add_trace(go.Scatter(
        x=holiday_data['date'],
        y=holiday_data['sales'],
        mode='markers',
        name='Holiday',
        marker=dict(size=12, color='red', symbol='diamond')
    ))
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Promotion Impact")
        promo_impact = historical_data.groupby('promotion')['sales'].mean()
        if len(promo_impact) > 1:
            impact = promo_impact[1] - promo_impact[0]
            st.success(f"Promotions boost sales by **${impact:,.0f}** on average!")
        
        fig2 = px.bar(
            x=['No Promotion', 'Promotion'],
            y=promo_impact.values,
            labels={'x': '', 'y': 'Average Sales ($)'},
            color=promo_impact.values,
            color_continuous_scale='Blues'
        )
        fig2.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        st.subheader("🎊 Holiday Impact")
        holiday_impact = historical_data.groupby('holiday')['sales'].mean()
        if len(holiday_impact) > 1:
            impact = holiday_impact[1] - holiday_impact[0]
            st.success(f"Holidays boost sales by **${impact:,.0f}** on average!")
        
        fig3 = px.bar(
            x=['Regular Day', 'Holiday'],
            y=holiday_impact.values,
            labels={'x': '', 'y': 'Average Sales ($)'},
            color=holiday_impact.values,
            color_continuous_scale='Reds'
        )
        fig3.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

# ============================================================================
# SINGLE PREDICTION PAGE
# ============================================================================
elif page == "🔮 Single Prediction":
    st.title("🔮 Make a Sales Prediction")
    st.markdown("### Select date and conditions to predict weekly sales")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📅 Prediction Settings")
        
        # Date picker
        min_date = historical_data['date'].max() + timedelta(days=1)
        max_date = min_date + timedelta(days=365)
        
        prediction_date = st.date_input(
            "Select prediction date:",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        col_a, col_b = st.columns(2)
        
        with col_a:
            promotion = st.checkbox("🎯 Promotion Active", value=False)
        
        with col_b:
            holiday = st.checkbox("🎊 Holiday Week", value=False)
        
        temperature = st.slider(
            "🌡️ Temperature (°C)",
            min_value=10.0,
            max_value=35.0,
            value=22.0,
            step=0.5
        )
        
        predict_button = st.button("🚀 Predict Sales", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("ℹ️ Quick Info")
        st.info(f"""
        **Historical Average:**  
        ${historical_data['sales'].mean():,.0f}
        
        **Date Range:**  
        {historical_data['date'].min().strftime('%Y-%m-%d')} to  
        {historical_data['date'].max().strftime('%Y-%m-%d')}
        
        **Total Records:**  
        {len(historical_data)} weeks
        """)
    
    if predict_button:
        # Make prediction
        future_df = pd.DataFrame({
            'ds': [pd.to_datetime(prediction_date)],
            'promotion': [1 if promotion else 0],
            'holiday': [1 if holiday else 0]
        })
        
        with st.spinner("🔮 Predicting..."):
            forecast = model.predict(future_df)
            
            predicted_sales = forecast['yhat'].values[0]
            lower_bound = forecast['yhat_lower'].values[0]
            upper_bound = forecast['yhat_upper'].values[0]
        
        # Display prediction
        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-box">
            <h2>Predicted Sales</h2>
            <h1 style="font-size: 48px; margin: 20px 0;">${predicted_sales:,.2f}</h1>
            <p style="font-size: 18px;">Confidence Interval: ${lower_bound:,.0f} - ${upper_bound:,.0f}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Impact analysis
        st.subheader("📊 Impact Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        # Base prediction (no promotion, no holiday)
        base_df = pd.DataFrame({
            'ds': [pd.to_datetime(prediction_date)],
            'promotion': [0],
            'holiday': [0]
        })
        base_forecast = model.predict(base_df)
        base_sales = base_forecast['yhat'].values[0]
        
        with col1:
            st.metric(
                "📊 Base Prediction",
                f"${base_sales:,.0f}",
                delta=None
            )
        
        if promotion:
            promo_impact = predicted_sales - model.predict(pd.DataFrame({
                'ds': [pd.to_datetime(prediction_date)],
                'promotion': [0],
                'holiday': [1 if holiday else 0]
            }))['yhat'].values[0]
            
            with col2:
                st.metric(
                    "🎯 Promotion Impact",
                    f"${abs(promo_impact):,.0f}",
                    delta=f"{promo_impact:+,.0f}"
                )
        
        if holiday:
            holiday_impact = predicted_sales - model.predict(pd.DataFrame({
                'ds': [pd.to_datetime(prediction_date)],
                'promotion': [1 if promotion else 0],
                'holiday': [0]
            }))['yhat'].values[0]
            
            with col3:
                st.metric(
                    "🎊 Holiday Impact",
                    f"${abs(holiday_impact):,.0f}",
                    delta=f"{holiday_impact:+,.0f}"
                )
        
        # Comparison chart
        st.subheader("📈 Prediction vs Historical Average")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=['Historical Average', 'Your Prediction'],
            y=[historical_data['sales'].mean(), predicted_sales],
            marker_color=['lightblue', 'darkblue'],
            text=[f"${historical_data['sales'].mean():,.0f}", f"${predicted_sales:,.0f}"],
            textposition='outside'
        ))
        
        fig.update_layout(
            height=400,
            yaxis_title="Sales ($)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Save prediction
        if st.button("💾 Save This Prediction"):
            pred_df = pd.DataFrame({
                'date': [prediction_date],
                'predicted_sales': [predicted_sales],
                'lower_bound': [lower_bound],
                'upper_bound': [upper_bound],
                'promotion': [1 if promotion else 0],
                'holiday': [1 if holiday else 0]
            })
            
            try:
                existing = pd.read_csv('my_predictions.csv')
                pred_df = pd.concat([existing, pred_df], ignore_index=True)
            except FileNotFoundError:
                pass
            
            pred_df.to_csv('my_predictions.csv', index=False)
            st.success("✅ Prediction saved to my_predictions.csv!")

# ============================================================================
# BATCH PREDICTIONS PAGE
# ============================================================================
elif page == "📊 Batch Predictions":
    st.title("📊 Batch Predictions")
    st.markdown("### Upload a CSV file to predict multiple dates at once")
    
    st.info("""
    **CSV Format Required:**
    - Columns: `date`, `promotion`, `holiday`
    - Date format: YYYY-MM-DD
    - Promotion/Holiday: 0 or 1
    """)
    
    # Sample template download
    sample_data = pd.DataFrame({
        'date': pd.date_range(start='2024-01-01', periods=10, freq='W'),
        'promotion': [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
        'holiday': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    })
    
    st.download_button(
        label="📥 Download Sample Template",
        data=sample_data.to_csv(index=False),
        file_name="prediction_template.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            batch_df['date'] = pd.to_datetime(batch_df['date'])
            
            st.success(f"✅ Loaded {len(batch_df)} dates for prediction")
            
            st.subheader("📋 Preview of Your Data")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("Predicting..."):
                    # Prepare for Prophet
                    future_df = batch_df.rename(columns={'date': 'ds'})
                    
                    # Make predictions
                    forecast = model.predict(future_df)
                    
                    # Combine results
                    results = pd.DataFrame({
                        'date': batch_df['date'],
                        'promotion': batch_df['promotion'],
                        'holiday': batch_df['holiday'],
                        'predicted_sales': forecast['yhat'].values,
                        'lower_bound': forecast['yhat_lower'].values,
                        'upper_bound': forecast['yhat_upper'].values
                    })
                
                st.success("✅ Predictions complete!")
                
                # Display results
                st.subheader("📊 Prediction Results")
                st.dataframe(results, use_container_width=True)
                
                # Summary stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Predictions", len(results))
                with col2:
                    st.metric("Average Predicted Sales", f"${results['predicted_sales'].mean():,.0f}")
                with col3:
                    st.metric("Total Predicted Revenue", f"${results['predicted_sales'].sum():,.0f}")
                
                # Chart
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['predicted_sales'],
                    mode='lines+markers',
                    name='Predicted Sales',
                    line=dict(color='blue', width=2)
                ))
                
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['upper_bound'],
                    mode='lines',
                    name='Upper Bound',
                    line=dict(width=0),
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=results['date'],
                    y=results['lower_bound'],
                    mode='lines',
                    name='Lower Bound',
                    fill='tonexty',
                    line=dict(width=0),
                    fillcolor='rgba(68, 68, 68, 0.2)'
                ))
                
                fig.update_layout(
                    title="Batch Predictions with Confidence Intervals",
                    xaxis_title="Date",
                    yaxis_title="Predicted Sales ($)",
                    height=500,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = results.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results",
                    data=csv,
                    file_name=f"batch_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    type="primary"
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your CSV has columns: date, promotion, holiday")

# ============================================================================
# MODEL ANALYTICS PAGE
# ============================================================================
elif page == "📈 Model Analytics":
    st.title("📈 Model Performance Analytics")
    
    # Load comparison data if available
    try:
        lr_pred = pd.read_csv('linear_regression_predictions.csv')
        prophet_pred = pd.read_csv('prophet_test_predictions.csv')
        
        st.success("✅ Model comparison data loaded")
        
        # Metrics comparison
        st.subheader("🏆 Model Comparison")
        
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        lr_mae = mean_absolute_error(lr_pred['actual_sales'], lr_pred['predicted_sales'])
        prophet_mae = mean_absolute_error(prophet_pred['actual_sales'], prophet_pred['predicted_sales'])
        
        lr_r2 = r2_score(lr_pred['actual_sales'], lr_pred['predicted_sales'])
        prophet_r2 = r2_score(prophet_pred['actual_sales'], prophet_pred['predicted_sales'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Linear Regression")
            st.metric("MAE", f"${lr_mae:,.2f}")
            st.metric("R² Score", f"{lr_r2:.4f}")
        
        with col2:
            st.markdown("### Prophet (Winner! 🏆)")
            st.metric("MAE", f"${prophet_mae:,.2f}", delta=f"-${lr_mae - prophet_mae:,.0f}")
            st.metric("R² Score", f"{prophet_r2:.4f}", delta=f"+{prophet_r2 - lr_r2:.4f}")
        
        # Predictions comparison chart
        st.subheader("📊 Actual vs Predicted")
        
        prophet_pred['date'] = pd.to_datetime(prophet_pred['date'])
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=prophet_pred['date'],
            y=prophet_pred['actual_sales'],
            mode='lines+markers',
            name='Actual Sales',
            line=dict(color='black', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=prophet_pred['date'],
            y=prophet_pred['predicted_sales'],
            mode='lines+markers',
            name='Prophet Predictions',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            height=500,
            xaxis_title="Date",
            yaxis_title="Sales ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        st.subheader("📉 Error Distribution")
        
        errors = prophet_pred['actual_sales'] - prophet_pred['predicted_sales']
        
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=errors,
            nbinsx=20,
            marker_color='lightblue',
            name='Errors'
        ))
        
        fig2.update_layout(
            height=400,
            xaxis_title="Prediction Error ($)",
            yaxis_title="Frequency",
            showlegend=False
        )
        
        st.plotly_chart(fig2, use_container_width=True)
        
        # Accuracy stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Error", f"${errors.mean():,.2f}")
        with col2:
            st.metric("Std Deviation", f"${errors.std():,.2f}")
        with col3:
            accuracy = (1 - abs(errors).mean() / prophet_pred['actual_sales'].mean()) * 100
            st.metric("Accuracy", f"{accuracy:.1f}%")
        
    except FileNotFoundError:
        st.warning("⚠️ Model comparison data not found. Run the comparison scripts first.")
        st.info("Run: `python 4_model_comparison.py`")

# ============================================================================
# SETTINGS PAGE
# ============================================================================
elif page == "⚙️ Settings":
    st.title("⚙️ Settings & Configuration")
    
    st.subheader("📁 Data Management")
    
    # Upload new data
    st.markdown("### Upload New Training Data")
    new_data = st.file_uploader("Upload sales_data.csv", type=['csv'])
    
    if new_data is not None:
        if st.button("💾 Save and Retrain Model"):
            with open('sales_data.csv', 'wb') as f:
                f.write(new_data.getbuffer())
            st.success("✅ Data uploaded! Retraining model...")
            st.cache_resource.clear()
            st.rerun()
    
    st.markdown("---")
    
    # Current data info
    st.subheader("📊 Current Dataset Info")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(historical_data))
    with col2:
        st.metric("Date Range", f"{(historical_data['date'].max() - historical_data['date'].min()).days} days")
    with col3:
        st.metric("Features", len(historical_data.columns))
    
    # Display current data
    if st.checkbox("Show raw data"):
        st.dataframe(historical_data, use_container_width=True)
    
    st.markdown("---")
    
    # Export options
    st.subheader("📤 Export Options")
    
    if st.button("📥 Download Current Data"):
        csv = historical_data.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # About
    st.subheader("ℹ️ About This Dashboard")
    st.info("""
    **Sales Prediction Dashboard v1.0**
    
    Built with:
    - Streamlit
    - Facebook Prophet
    - Plotly
    - Python 3.13
    
    Created for sales forecasting and business intelligence.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; padding: 20px;">
    Sales Prediction Dashboard | Powered by AI | Built with Streamlit
</div>
""", unsafe_allow_html=True)
