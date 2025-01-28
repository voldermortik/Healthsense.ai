import httpx
import pandas as pd
from crewai import Agent
from crewai.tools import BaseTool
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from constants import MODEL_NAME, HOSPITAL_INFO_FILE_PATH


class HospitalComparisonAgent:
    """
   An agent designed to analyze hospital data using pandas dataframe and return human-readable output.

   Attributes:
       hospital_info_agent (Agent): An agent configured to interact with pandas dataframes and
       return hospital comparisons.
   """
    def __init__(self):
        """
        Initializes the HospitalComparisonAgent with a role and tool for analyzing hospital data.
        """
        self.hospital_info_agent = Agent(
            role='Hospital Information Analyst',
            goal='Analyze hospital data using pandas dataframe and return relevant human readable output',
            backstory='Expert at analyzing complex datasets using pandas dataframe',
            tools=[PandasTool(result_as_answer=True)],
            verbose=True
        )

class PandasTool(BaseTool):
    """
    A tool for querying a pandas dataframe, analyzing data, and returning human-readable output regarding hospitals.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool's purpose and capabilities.
    """
    name: str = "pandas_tool"
    description: str = "Query pandas dataframe, analyze data and return relevant human readable output"

    def _run(self, query: str) -> str:
        """
        Executes a query on a pandas dataframe containing hospital data and returns a human-readable response.

        Args:
            query (str): The query string to be executed.

        Returns:
            str: The result of the query as a human-readable output.
        """
        llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME, max_tokens=500,
                              http_client=httpx.Client(verify=False))
        df = pd.read_csv(HOSPITAL_INFO_FILE_PATH)

        system_message = SystemMessagePromptTemplate.from_template(
            """
            You are a highly skilled healthcare assistant with expertise in comparing hospitals. 
            Your task is to assess various hospitals based on a user's specific conditions, preferences, and needs. 
            You will evaluate hospitals considering factors such as medical specialties, patient reviews, location, cost, accessibility, facilities, 
            and the availability of treatment for specific conditions.

            When comparing hospitals, follow these guidelines:

            - Condition-Specific Comparison: Focus on the hospitals' expertise in treating the user's specific health condition 
            (e.g., heart disease, cancer, etc.).
            - Hospital Features: Include details about the hospital's reputation, technology, facilities, specialized care, and any awards or 
            recognitions.
            - Location and Accessibility: Consider the proximity to the user’s location and the convenience of travel.
            - Cost and Insurance: Compare the cost of treatment and insurance coverage options offered by the hospitals.
            - Patient Feedback: Analyze reviews and ratings to gauge patient satisfaction and outcomes.
            - Personalized Recommendation: Provide a clear, personalized suggestion based on the user’s priorities, whether they are medical 
            expertise, convenience, or cost.


            Use "Hospital Type" column to look for good facilities of each hospital.
            CAREFULLY look at each column name to understand what to output.
            """
        )

        prompt = ChatPromptTemplate.from_messages([system_message])
        pandas_agent = create_pandas_dataframe_agent(llm, df, prompt=prompt, verbose=True,
                                                          allow_dangerous_code=True,
                                                          agent_type=AgentType.OPENAI_FUNCTIONS)
        result = pandas_agent.invoke(query)
        print(result)
        return  result['output']

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported")
