from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from langchain_groq import ChatGroq
import os
from langchain_openai import ChatOpenAI



# Tool imports
from tools.newstool import StockNewsScraperTool
from tools.stockprice import FinancialDataFetcherTool
from tools.mpt import PortfolioOptimizationTool
from tools.searchtool import SearchTool

@CrewBase
class Synergy():
    """Synergy crew for stock analysis"""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        # Initialize tools
        self.news_tool = StockNewsScraperTool()
        self.financial_tool = FinancialDataFetcherTool()
        self.portfolio_tool = PortfolioOptimizationTool()
        self.search_tool = SearchTool()

        
        self.groq_llm = ChatGroq(
            temperature=0,
            model_name="groq/mixtral-8x7b-32768",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=4000
        )
        # self.groq_llm = ChatOpenAI(
        #     temperature=0,
        #     model="gpt-4-turbo",  # Use 'gpt-3.5-turbo' for lower cost
        #     openai_api_key=os.getenv("OPENAI_API_KEY"),
        #     max_tokens=4000
        # )

    # ------------------ Agents ------------------
    @agent
    def data_extraction_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['data_extraction_agent'],
            llm=self.groq_llm,
            tools=[self.news_tool, self.financial_tool, self.search_tool],
            verbose=True,
            allow_delegation=False,
            max_iter=5,max_rpm=29
        )

    @agent
    def stock_recommendation_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['stock_recommendation_agent'],
            llm=self.groq_llm,
            tools=[self.search_tool],
            verbose=True,
            allow_delegation=False,
            max_iter=3,max_rpm=29
        )

    @agent
    def investment_strategy_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['investment_strategy_agent'],
            llm=self.groq_llm,
            tools=[self.portfolio_tool],
            verbose=True,
            allow_delegation=True,
            max_iter=5,max_rpm=29
        )

    # ------------------ Tasks ------------------
    @task
    def data_extraction_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_extraction_task'],
            agent=self.data_extraction_agent(),
            output_file="extracted_data.md"
        )

    @task
    def stock_recommendation_task(self) -> Task:
        return Task(
            config=self.tasks_config['stock_recommendation_task'],
            agent=self.stock_recommendation_agent(),
            context=[self.data_extraction_task()],
            output_file="recommendation_report.md"
        )

    @task
    def investment_strategy_task(self) -> Task:
        return Task(
            config=self.tasks_config['investment_strategy_task'],
            agent=self.investment_strategy_agent(),
            context=[self.stock_recommendation_task()],
            output_file="investment_strategy.md"
        )

    @crew
    def crew(self) -> Crew:
        """Assemble the crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            manager_llm=self.groq_llm,  # Critical Groq integration
            verbose=True,
            memory=True
        )