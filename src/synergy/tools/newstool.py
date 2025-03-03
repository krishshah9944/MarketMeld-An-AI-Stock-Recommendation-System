from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import os
import re
import requests
from dotenv import load_dotenv

# Load environment variables (ensure your .env file has NEWSDATA_API_KEY)
load_dotenv()

newsdata_api_key = os.getenv("NEWSDATA_API_KEY")
if not newsdata_api_key:
    print("NEWSDATA_API_KEY not found in environment!")
else:
    print("NEWSDATA_API_KEY loaded successfully.")

# Input schema for scraping stock news
class StockNewsScraperInput(BaseModel):
    stock_name: str = Field(
        ...,
        description="Name or symbol of the stock to retrieve news for (e.g., 'Tata Motors')."
    )

# Helper function to check for a whole-word match using regex
def contains_word(text: str, word: str) -> bool:
    # If text is None or empty, return False immediately
    if not text:
        return False
    pattern = r'\b' + re.escape(word) + r'\b'
    return re.search(pattern, text, re.IGNORECASE) is not None

# Crew AI tool for fetching news about a given stock from Newsdata.io
class StockNewsScraperTool(BaseTool):
    name: str = "Stock News Scraper"
    description: str = (
        "Fetches recent news articles related to a given stock using the Newsdata.io API."
    )
    args_schema: Type[BaseModel] = StockNewsScraperInput

    def _run(self, stock_name: str) -> dict:
        """
        Scrapes recent news headlines, descriptions, and links for the specified stock
        using Newsdata.io. Returns a dictionary containing a list of news articles.
        """
        # Refine the query by appending additional keywords to steer the results toward stock news.
        query = f"{stock_name} stock news"
        
        # Build the request URL with query, API key, and optional filters (e.g., language)
        url = (
            f"https://newsdata.io/api/1/news?"
            f"apikey={newsdata_api_key}&q={query}&language=en"
        )
        
        # Make the HTTP GET request
        response = requests.get(url)
        if response.status_code != 200:
            return {"error": f"Request failed with status code {response.status_code}"}
        
        # Parse JSON response from Newsdata.io
        data = response.json()
        articles = data.get("results", [])
        
        # Build a structured list of articles
        parsed_articles = []
        for item in articles:
            news_item = {
                "title": item.get("title", ""),
                "link": item.get("link", ""),
                "description": item.get("description", ""),
                "pubDate": item.get("pubDate", ""),
                "source": item.get("source_id", "")
            }
            parsed_articles.append(news_item)
        
        # Post-process: filter articles that don't explicitly mention the stock name as a whole word
        filtered_articles = [
            article for article in parsed_articles
            if contains_word(article.get("title", ""), stock_name)
            or contains_word(article.get("description", ""), stock_name)
        ]
        
        return {"stock_news": filtered_articles}

# Main block to test the tool

