import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
from datetime import datetime, timedelta
import plotly.express as px
import joblib
import os

def load_data(file):
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Error Loading File: {e}")
        return None

def prepare_data(df, date_column, value_column):
    try:
        # Convert date column to datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Convert value column to numeric, forcing errors to NaN
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna()
        
        # Check if we have any data left after cleaning
        if len(df) == 0:
            st.error("No valid data remaining after cleaning. Please check your input data.")
            return None
            
        # Set Date as Index
        df = df.set_index(date_column)
        # Select only the value column
        df = df[[value_column]]
        # Sort Index
        df = df.sort_index()
        return df
    except Exception as e:
        st.error(f"Error preparing data: {e}")
        return None

def create_train_test_split(df, test_size=0.2):
    split_point = int(len(df) * (1 - test_size))
    train = df[:split_point]
    test = df[split_point:]
    return train, test

def train_model(train_data, seasonal_periods=7):
    model = ExponentialSmoothing(
        train_data,
        seasonal_periods=seasonal_periods,
        trend='add',
        seasonal='add',
        use_boxcox=True
    )
    model_fit = model.fit()
    return model_fit

def make_forecast(model, forecast_steps):
    forecast = model.forecast(steps=forecast_steps)
    return forecast

def plot_results(original_data, train_data, test_data, forecast_data):
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(
        x=train_data.index,
        y=train_data.values.flatten(),
        name='Training Data',
        mode='lines'
    ))

    # Plot test data
    if test_data is not None:
        fig.add_trace(go.Scatter(
            x=test_data.index,
            y=test_data.values.flatten(),
            name='Test Data',
            mode='lines'
        ))

    # Plot forecast
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data.values,
        name='Forecast',
        mode='lines',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title='Time Series Forecast',
        xaxis_title='Date',
        yaxis_title='Value',
        hovermode='x'
    )

    return fig

def plot_components(data):
    decomposition = seasonal_decompose(data, period=7)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.values.flatten(), name='Original'))
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.trend, name='Trend'))
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.seasonal, name='Seasonal'))
    fig.add_trace(go.Scatter(x=data.index, y=decomposition.resid, name='Residual'))
    
    fig.update_layout(
        title='Time Series Components',
        xaxis_title='Date',
        yaxis_title='Value',
        height=800
    )
    
    return fig

def main():
    st.title('Time Series Forecasting App')
    st.write("""
    This app performs time series forecasting using Holt-Winters' method.
    Upload your data and configure the parameters to get started.
    """)

    # File upload
    uploaded_file = st.file_uploader("Upload your time series data (CSV)", type=['csv'])
    
    if uploaded_file is not None:
        # Load data
        df = load_data(uploaded_file)
        
        if df is not None:
            st.write("Preview of uploaded data:")
            st.write(df.head())
            
            # Data preparation parameters
            col1, col2 = st.columns(2)
            with col1:
                date_column = st.selectbox('Select date column', df.columns)
            with col2:
                value_column = st.selectbox('Select value column', df.columns)
            
            # Prepare data
            prepared_data = prepare_data(df, date_column, value_column)
            
            if prepared_data is not None:
                # Model parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    test_size = st.slider('Test set size (%)', 0, 50, 20)
                with col2:
                    seasonal_periods = st.number_input('Seasonal Periods', 
                                                     min_value=1, 
                                                     value=7)
                with col3:
                    forecast_steps = st.number_input('Forecast Steps', 
                                                   min_value=1, 
                                                   value=30)
                
                if st.button('Train Model and Forecast'):
                    # Split data
                    train_data, test_data = create_train_test_split(
                        prepared_data, 
                        test_size=test_size/100
                    )
                    
                    with st.spinner('Training model...'):
                        # Train model
                        model = train_model(train_data, seasonal_periods)
                        
                        # Make forecast
                        forecast = make_forecast(model, forecast_steps)
                        
                        # Plot results
                        fig = plot_results(prepared_data, train_data, 
                                         test_data, forecast)
                        st.plotly_chart(fig)
                        
                        # Plot components
                        st.subheader('Time Series Components')
                        components_fig = plot_components(prepared_data)
                        st.plotly_chart(components_fig)
                        
                        # Display metrics
                        if len(test_data) > 0:
                            mse = np.mean((test_data.values - 
                                         model.forecast(len(test_data)).values)**2)
                            rmse = np.sqrt(mse)
                            st.write(f'Root Mean Square Error (RMSE): {rmse:.2f}')
                        
                        # Option to download the model
                        if not os.path.exists('models'):
                            os.makedirs('models')
                        model_path = 'models/time_series_model.pkl'
                        joblib.dump(model, model_path)
                        
                        with open(model_path, 'rb') as f:
                            st.download_button(
                                label='Download trained model',
                                data=f,
                                file_name='time_series_model.pkl',
                                mime='application/octet-stream'
                            )

if __name__ == '__main__':
    main()        