import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(42)

# Generate dates for 2 years of weekly data
start_date = datetime(2022, 1, 1)
dates = [start_date + timedelta(weeks=x) for x in range(104)]  # 104 weeks = 2 years

# Generate components for the time series
def generate_sales_data():
    # Base trend (growing over time)
    trend = np.linspace(1000, 1500, 104)
    
    # Seasonal components
    # Yearly seasonality (higher in summer, lower in winter)
    yearly_seasonality = 200 * np.sin(np.arange(104) * (2 * np.pi / 52))
    
    # Quarterly seasonality (e.g., business quarters)
    quarterly_seasonality = 100 * np.sin(np.arange(104) * (2 * np.pi / 13))
    
    # Monthly promotions (spike every 4 weeks)
    monthly_effect = 150 * (np.arange(104) % 4 == 0).astype(float)
    
    # Random noise
    noise = np.random.normal(0, 50, 104)
    
    # Combine all components
    sales = trend + yearly_seasonality + quarterly_seasonality + monthly_effect + noise
    
    # Ensure no negative values
    sales = np.maximum(sales, 0)
    
    return sales.round(2)

# Generate the data
sales = generate_sales_data()

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'weekly_sales': sales
})

# Add some metadata columns
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['week'] = df['date'].dt.isocalendar().week

# Save to CSV
df.to_csv('sample_weekly_sales_data.csv', index=False)

# Display first few rows and data info
print("\nFirst few rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())