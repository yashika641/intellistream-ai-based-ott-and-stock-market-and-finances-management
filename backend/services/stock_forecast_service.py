import random

def forecast_stock(company_name: str, days: int):
    # Dummy forecast: generate fake prices
    base_price = 100 + random.randint(-10, 10)
    forecast = [round(base_price + random.uniform(-5, 5), 2) for _ in range(days)]
    return forecast
