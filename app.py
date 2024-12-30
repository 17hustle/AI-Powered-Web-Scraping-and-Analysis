import streamlit as st
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from crewai_tools import ScrapeWebsiteTool, FileReadTool
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
Model = 'llama-3.1-70b-versatile'
llm = ChatGroq(model=Model, api_key=api_key)

# Streamlit App
st.title("AI-Powered Web Scraping and Analysis")
st.markdown("""
This app performs two tasks:
1. Scrapes business-related data from a website.
2. Analyzes industry trends and proposes AI/ML use cases for the company.
""")

# Inputs
website_url = st.text_input("Enter the Website URL", value="https://aiplanet.com/")

# Initialize session state for persistence
if "scraped_data" not in st.session_state:
    st.session_state.scraped_data = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

# Web Scraping
if st.button("Start Web Scraping"):
    st.info("Initializing web scraping...")

    # Web Scraper Agent
    web_scrape_tool = ScrapeWebsiteTool(website_url=website_url)

    web_scraper_agent = Agent(
        role='Web Scraper',
        goal='Scrape specific business-related data from the website.',
        backstory='''You are a focused web scraper specializing in extracting information about
                     a company’s industry, key offerings, and strategic focus areas.''',
        tools=[web_scrape_tool],
        verbose=True,
        llm=llm
    )

    # Define task
    web_scraper_task = Task(
        description='Extract the industry of the company, its key offerings, and strategic focus areas from the website.',
        expected_output='Details about the company’s industry, key offerings, and strategic focus areas.',
        agent=web_scraper_agent,
        output_file='data_specific.txt'
    )

    # Assemble crew
    crew = Crew(
        agents=[web_scraper_agent],
        tasks=[web_scraper_task],
        verbose=2,
    )

    # Execute task
    result = crew.kickoff()
    st.session_state.scraped_data = result  # Store the result in session state
    st.success("Web scraping completed!")
    st.text_area("Scraped Data", result, height=300)

# Trend Analysis
if st.session_state.scraped_data:
    if st.button("Analyze Trends"):
        st.info("Starting analysis...")

        analyze_trends_tool = FileReadTool(input_data=st.session_state.scraped_data)

        # Trend Analysis Agent
        trend_analysis_agent = Agent(
            role='Industry Analyst',
            goal='Analyze industry trends and propose relevant AI/ML/GenAI use cases for the company.',
            backstory='''You are an expert in industry analysis and identifying how emerging technologies like 
                         AI, ML, and GenAI can transform business processes. Your goal is to provide actionable 
                         insights and recommendations for leveraging these technologies within the company.''',
            tools=[analyze_trends_tool],
            verbose=True,
            llm=llm
        )

        # Define trend analysis task
        trend_analysis_task = Task(
            description='''Analyze the industry trends and propose actionable use cases where the company can leverage AI, ML, and GenAI technologies.''',
            expected_output='''Comprehensive industry trends and use cases for AI, ML, and GenAI.''',
            agent=trend_analysis_agent,
            output_file='analysis_results.txt'
        )

        # Assemble crew for analysis
        crew = Crew(
            agents=[trend_analysis_agent],
            tasks=[trend_analysis_task],
            verbose=2,
        )

        # Execute analysis
        analysis_result = crew.kickoff()
        st.session_state.analysis_result = analysis_result  # Store the result in session state
        st.success("Trend analysis completed!")
        st.text_area("Analysis Results", analysis_result, height=400)
