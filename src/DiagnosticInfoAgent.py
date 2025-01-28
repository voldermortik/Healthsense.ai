import httpx
import pandas as pd
from crewai import Agent
from crewai.tools import BaseTool
from langchain.agents.agent_types import AgentType
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

from constants import MODEL_NAME, DIAGNOSTIC_INFO_FILE_PATH


class DiagnosticInfoAgent:
    """
    A diagnostic information agent designed to analyze diagnostic tests and packages from hospital data.

    Attributes:
        diagnostic_info_agent (Agent): An agent configured with tools and logic to process
        hospital diagnostic data.
    """
    def __init__(self):
        """
        Initializes the DiagnosticInfoAgent with a diagnostic information analysis role and tool.
        """
        self.diagnostic_info_agent = Agent(
            role='Diagnostic Information Finder',
            goal='Analyze diagnostic tests and packages from hospital data',
            backstory='Expert at analyzing complex datasets using csv',
            tools=[DiagnosticTool(result_as_answer=True)],
            verbose=True
        )

class DiagnosticTool(BaseTool):
    """
    A tool for querying and analyzing data from a pandas DataFrame.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool's functionality.
    """
    name: str = "pandas_tool"
    description: str = "Query pandas dataframe and analyze data"

    def _run(self, query: str) -> str:
        """
        Executes a query against a pandas DataFrame to analyze diagnostic information.

        Args:
            query (str): The query string specifying the analysis or information required.

        Returns:
            str: The output of the analysis as a string.
        """
        llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME, max_tokens=500,
                              http_client=httpx.Client(verify=False))
        diagnostic_agent_system_message = SystemMessagePromptTemplate.from_template(
            """
            You are a highly skilled healthcare assistant with expertise in suggesting health screening tests and packagaes. 
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

            CAREFULLY look at each column name to understand what to output.
            """
        )

        diagnostic_agent = create_pandas_dataframe_agent(llm,
                                                          pd.read_csv(DIAGNOSTIC_INFO_FILE_PATH),
                                                          prompt=ChatPromptTemplate.from_messages(
                                                              [diagnostic_agent_system_message]),
                                                          verbose=True,
                                                          allow_dangerous_code=True,
                                                          agent_type=AgentType.OPENAI_FUNCTIONS)
        return diagnostic_agent.invoke(query)['output']

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported")