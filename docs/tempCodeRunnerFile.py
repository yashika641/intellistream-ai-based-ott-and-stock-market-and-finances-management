import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()

def generate_synthetic_stock_data(start_date='2018-01-01', end_date='2025-6-28'):
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    
    prices = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    volumes = []
    
    base_price = 100.0
    price = base_price
    
    for i, current_date in enumerate(dates):
        # Weekly seasonality effect: prices tend to rise on Mondays, drop on Fridays
        day_of_week = current_date.dayofweek  # 0=Mon, ..., 4=Fri
        weekly_seasonality = 0
        if day_of_week == 0:
            weekly_seasonality = random.uniform(0.002, 0.01)  # positive bump on Mondays
        elif day_of_week == 4:
            weekly_seasonality = random.uniform(-0.01, -0.002)  # dip on Fridays
        
        # Monthly seasonality: slight price increase towards end of month
        day_of_month = current_date.day
        monthly_seasonality = 0.005 if day_of_month > 20 else 0
        
        # Random daily fluctuation
        daily_change = random.uniform(-0.03, 0.03)
        
        # Combine effects
        total_change = daily_change + weekly_seasonality + monthly_seasonality
        
        price = price * (1 + total_change)
        price = max(price, 1)  # no negative or zero prices
        
        # Simulate OHLC (open, high, low, close)
        open_price = price * random.uniform(0.995, 1.005)
        close_price = price
        high_price = max(open_price, close_price) * random.uniform(1.00, 1.02)
        low_price = min(open_price, close_price) * random.uniform(0.98, 1.00)
        
        # Volume: base volume fluctuates + some random noise
        base_volume = 1_000_000
        volume = int(base_volume * random.uniform(0.5, 1.5))
        
        open_prices.append(round(open_price, 2))
        high_prices.append(round(high_price, 2))
        low_prices.append(round(low_price, 2))
        close_prices.append(round(close_price, 2))
        volumes.append(volume)
    
    df = pd.DataFrame({
        'date': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    return df

# Generate and save synthetic stock data
df_stock = generate_synthetic_stock_data()
print(df_stock.head())

df_stock.to_csv('synthetic_stock_data_enhanced.csv', index=False)
print("Enhanced synthetic stock data saved as 'synthetic_stock_data_enhanced.csv'")
