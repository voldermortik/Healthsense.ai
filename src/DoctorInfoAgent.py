import httpx
from crewai import Agent
from crewai.tools import BaseTool
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from constants import MODEL_NAME


class DoctorInfoAgent:
    """
    An agent to analyze doctor availability data and handle SQL queries related to doctor scheduling.

    Attributes:
        doctor_slots_agent (Agent): An agent configured to interact with SQL databases for doctor
        slot analysis.
    """
    def __init__(self):
        """
        Initializes the DoctorInfoAgent with a role and tool for analyzing doctor availability data.
        """
        self.doctor_slots_agent = Agent(
            role='Doctor Availability Checker and Slot Booking',
            goal='Analyze doctor availability data and books slots if asked',
            backstory='Expert at analyzing complex datasets using SQL',
            tools=[SlotsQueryTool(result_as_answer=True)],
            verbose=True
            )

class SlotsQueryTool(BaseTool):
    """
    A tool for running SQL queries and analyzing database data related to doctor availability.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool's purpose and capabilities.
    """
    name: str = "sql_query_tool"
    description: str = "Run SQL queries and analyze database data"

    def _run(self, query: str) -> str:
        """
        Executes a SQL query to analyze doctor availability data.

        Args:
            query (str): The SQL query string to be executed.

        Returns:
            str: The output of the query execution as a string.
        """
        llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME, max_tokens=500,
                              http_client=httpx.Client(verify=False))

        template = '''
            Answer the following questions as best you can. You have access to the following tools:

            {tools}
    
            Use the following format:
    
            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question
    
            Use like operator with lowercase when matching a name. 
            When a user is asking to book slots for any dr, STRICTLY Delete the corresponding row from the table.
            Begin!
    
            Question: {input}
            Thought:{agent_scratchpad}'''
        doctor_info_agent = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=SQLDatabase.from_uri("sqlite:///appointments.db"), llm=llm),
            prompt=PromptTemplate.from_template(template),
            verbose=True,
            handle_parsing_errors=True,
        )
        return doctor_info_agent.invoke(query)['output']

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported")
