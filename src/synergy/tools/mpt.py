from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type, List
import cvxpy as cp
import numpy as np

# Input schema now requires the stocks to be passed by the agent
class PortfolioOptimizationInput(BaseModel):
    capital: float = Field(..., description="Total capital available for investment.")
    risk_tolerance: str = Field(..., description="Risk tolerance level: 'conservative', 'moderate', or 'aggressive'.")
    stocks: List[str] = Field(..., description="List of recommended stock symbols passed by the agent.")

def optimize_portfolio(expected_returns: np.ndarray, cov_matrix: np.ndarray, target_return: float) -> np.ndarray:
    """
    Solves a mean–variance optimization problem:
      minimize portfolio variance subject to achieving at least the target return.
    """
    n = len(expected_returns)
    w = cp.Variable(n)
    portfolio_return = expected_returns @ w
    portfolio_variance = cp.quad_form(w, cov_matrix)
    objective = cp.Minimize(portfolio_variance)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        portfolio_return >= target_return
    ]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    return w.value

# Crew AI tool for portfolio optimization that now accepts a list of stocks from the agent
class PortfolioOptimizationTool(BaseTool):
    name: str = "Portfolio Optimization Tool"
    description: str = (
        "Optimizes portfolio allocation given total capital, risk tolerance, and a list of stocks. "
        "The agent passes the recommended stocks, and the tool calculates optimal weights and dollar allocations "
        "using a mean–variance (Modern Portfolio Theory) approach."
    )
    args_schema: Type[BaseModel] = PortfolioOptimizationInput

    def _run(self, capital: float, risk_tolerance: str, stocks: List[str]) -> dict:
        # In a production system, you would fetch real market data for these stocks.
        # For demonstration, we create dummy expected returns and a covariance matrix for the provided stocks.
        num_stocks = len(stocks)
        dummy_expected_returns = np.full((num_stocks,), 0.12)  # Example: all stocks have a 12% expected return.
        # Using a simple diagonal covariance matrix (replace with real correlations and variances)
        dummy_cov_matrix = 0.005 * np.eye(num_stocks)
        
        # Map risk tolerance to a target return (adjust thresholds as needed)
        risk_target_map = {
            "conservative": 0.10,
            "moderate": 0.12,
            "aggressive": 0.15
        }
        target_return = risk_target_map.get(risk_tolerance.lower(), 0.12)
        
        # Optimize portfolio allocation
        weights = optimize_portfolio(dummy_expected_returns, dummy_cov_matrix, target_return)
        
        # Calculate dollar allocations based on provided capital
        allocations = {stock: round(float(weight) * capital, 2) for stock, weight in zip(stocks, weights)}
        formatted_weights = {stock: round(float(weight), 4) for stock, weight in zip(stocks, weights)}
        
        return {
            "capital": capital,
            "risk_tolerance": risk_tolerance,
            "stocks": stocks,
            "optimal_weights": formatted_weights,
            "allocations": allocations
        }


