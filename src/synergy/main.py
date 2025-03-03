#!/usr/bin/env python
import sys
import warnings
import os
from dotenv import load_dotenv
from synergy.crew import Synergy

# Load the .env file from the correct directory
dotenv_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(dotenv_path)

print("Groq API Key:", os.getenv("GROQ_API_KEY"))
print("OpenAI API Key:", os.getenv("OPENAI_API_KEY"))

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def validate_env():
    required = ['GROQ_API_KEY', 'NEWSDATA_API_KEY', 'FMP_API_KEY', 'SERPER_API_KEY']
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        exit(1)

def invest():
    """
    Run the Investment Strategy flow.
    
    Prompts the user for:
      - Stock symbol for recommendation,
      - Investment amount,
      - Risk appetite (conservative, moderate, or aggressive).
    
    The crew then uses these inputs to:
      1. Run the Stock Recommendation Task,
      2. Aggregate the recommendation with investment parameters,
      3. Run the Investment Strategy Task to generate an optimized strategy.
    """
    validate_env()
    
    stock = input("Enter stock symbol (e.g., TSLA): ").strip().upper()
    try:
        amount = float(input("Enter total investment amount: "))
    except ValueError:
        print("Invalid input for investment amount. Please enter a numeric value.")
        return
    risk = input("Enter risk (conservative/moderate/aggressive): ").strip().lower()
    
    # Build input dictionary for the crew
    inputs = {
        "stock": stock,
        "amount": amount,
        "risk": risk
    }
    
    # Instantiate the crew and run the entire workflow.
    # This will automatically run the stock recommendation task (if needed) and then the investment strategy task.
    crew_instance = Synergy().crew()
    final_output = crew_instance.kickoff(inputs=inputs)
    
    print("\n💼 Final Investment Strategy Output:")
    print(final_output)

def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "stock": input("Enter stock symbol for training: ").strip().upper(),
        "amount": float(input("Enter investment amount for training: ")),
        "risk": input("Enter risk (conservative/moderate/aggressive): ").strip().lower()
    }
    try:
        Synergy().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Synergy().crew().replay(task_id=sys.argv[1])
    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and return the results.
    """
    inputs = {
        "stock": input("Enter stock symbol for testing: ").strip().upper(),
        "amount": float(input("Enter investment amount for testing: ")),
        "risk": input("Enter risk (conservative/moderate/aggressive): ").strip().lower()
    }
    try:
        Synergy().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)
    except Exception as e:
        raise Exception(f"An error occurred while testing the crew: {e}")

if __name__ == "__main__":
    invest()
