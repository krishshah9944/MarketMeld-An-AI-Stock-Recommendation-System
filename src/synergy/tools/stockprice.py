from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import os
import requests
import datetime
from dotenv import load_dotenv

# Load environment variables (ensure your .env file has FMP_API_KEY)
load_dotenv()

FMP_API_KEY = os.getenv("FMP_API_KEY")
if not FMP_API_KEY:
    print("FMP_API_KEY not found in environment!")
else:
    print("FMP_API_KEY loaded successfully.")

# Input schema for fetching comprehensive financial data
class FinancialDataFetcherInput(BaseModel):
    company_name: str = Field(
        ...,
        description="Company name to fetch data for (e.g., 'Tesla', 'Microsoft', 'Apple')."
    )

def get_stock_symbol(company_name: str) -> str:
    """Convert a company name to its stock symbol using FMP API."""
    url = f"https://financialmodelingprep.com/api/v3/search?query={company_name}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            # Return first matching symbol
            return data[0].get('symbol')
        else:
            print("❌ No matching company found.")
            return None
    else:
        print(f"❌ Error fetching stock symbol! Status Code: {response.status_code}")
        print(f"🔹 Response: {response.text}")
        return None

def get_financial_data(symbol: str) -> dict:
    """Fetch financial statements (income statement, balance sheet, cash flow) for the given symbol."""
    endpoints = {
        "income_statement": f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?limit=1&apikey={FMP_API_KEY}",
        "balance_sheet": f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?limit=1&apikey={FMP_API_KEY}",
        "cash_flow": f"https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?limit=1&apikey={FMP_API_KEY}"
    }
    
    financial_data = {}
    for key, url in endpoints.items():
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                financial_data[key] = data
                print(f"✅ {key.replace('_', ' ').title()} fetched successfully!")
            else:
                print(f"❌ No data available for {key.replace('_', ' ').title()}.")
                financial_data[key] = None
        else:
            print(f"❌ Error fetching {key.replace('_', ' ').title()}. Status Code: {response.status_code}")
            print(f"🔹 Response: {response.text}")
            financial_data[key] = None
    return financial_data

def get_stock_price_data(symbol: str) -> dict:
    """Fetch the latest stock price data using the FMP quote endpoint."""
    url = f"https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={FMP_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data:
            print("✅ Stock Price data fetched successfully!")
            # Return the first object from the quote data
            return data[0]
        else:
            print("❌ No stock price data available.")
            return None
    else:
        print(f"❌ Error fetching stock price data. Status Code: {response.status_code}")
        print(f"🔹 Response: {response.text}")
        return None

def get_last_5_days_prices(symbol: str) -> list:
    """Fetch historical stock prices (last 5 days) for the given symbol."""
    today = datetime.date.today()
    # Subtracting 10 days to cover weekends and holidays
    start_date = today - datetime.timedelta(days=10)
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}"
        f"?from={start_date}&to={today}&apikey={FMP_API_KEY}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data and "historical" in data:
            historical = data["historical"]
            # Assuming the API returns data in descending order (latest first)
            last_5_days = historical[:5]
            print("✅ Last 5 days price data fetched successfully!")
            return last_5_days
        else:
            print("❌ No historical trend data available.")
            return None
    else:
        print(f"❌ Error fetching historical trend data. Status Code: {response.status_code}")
        print(f"🔹 Response: {response.text}")
        return None

# Crew AI tool for fetching comprehensive financial and stock data
class FinancialDataFetcherTool(BaseTool):
    name: str = "Comprehensive Financial Data Fetcher"
    description: str = (
        "Fetches financial statements (income statement, balance sheet, cash flow), "
        "latest stock price (including one day return), and historical trend data (last 5 days) "
        "for a given company using the FMP API."
    )
    args_schema: Type[BaseModel] = FinancialDataFetcherInput

    def _run(self, company_name: str) -> dict:
        # Convert company name to stock symbol
        stock_symbol = get_stock_symbol(company_name)
        if not stock_symbol:
            return {"error": "No matching company found or error fetching stock symbol."}
        
        # Fetch financial data
        financials = get_financial_data(stock_symbol)
        
        # Fetch latest stock price data
        price_data = get_stock_price_data(stock_symbol)
        
        # Extract one day return if available (FMP usually returns "changesPercentage")
        one_day_return = None
        if price_data and "changesPercentage" in price_data:
            one_day_return = price_data["changesPercentage"]
        
        # Fetch historical trend data for last 5 days
        last_5_days_prices = get_last_5_days_prices(stock_symbol)
        
        return {
            "company_name": company_name,
            "stock_symbol": stock_symbol,
            "financial_data": financials,
            "stock_price": price_data,
            "one_day_return": one_day_return,
            "last_5_days_prices": last_5_days_prices
        }

#