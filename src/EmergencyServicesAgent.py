import httpx
from crewai import Agent
from crewai.tools import BaseTool
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain.prompts import PromptTemplate
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

from constants import MODEL_NAME


class EmergencyServicesAgent:
    """
    An agent designed to analyze emergency services availability data using SQL queries.

    Attributes:
        emergency_agent (Agent): An agent configured to interact with SQL databases for
        analyzing emergency services data.
    """
    def __init__(self):
        """
        Initializes the EmergencyServicesAgent with a role and tool for analyzing emergency services data.
        """
        self.emergency_agent = Agent(
            role='Emergency Information Finder',
            goal='Analyze emergency services availability data',
            backstory='Expert at analyzing complex datasets using SQL',
            tools=[EmergencyQueryTool(result_as_answer=True)],
            verbose=True
        )

class EmergencyQueryTool(BaseTool):
    """
    A tool for running SQL queries and analyzing database data related to emergency services availability.

    Attributes:
        name (str): Name of the tool.
        description (str): Description of the tool's purpose and capabilities.
    """
    name: str = "sql_query_tool"
    description: str = "Run SQL queries and analyze database data"

    def _run(self, query: str) -> str:
        """
        Executes a SQL query to analyze emergency services availability data.

        Args:
            query (str): The SQL query string to be executed.

        Returns:
            str: The output of the query execution as a string.
        """
        llm = ChatOpenAI(temperature=0, model_name=MODEL_NAME, max_tokens=500,
                              http_client=httpx.Client(verify=False))

        template = '''Answer the following questions as best you can. You have access to the following tools:

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

                Use like operator with lowercase when matching a name. When user is asking to book slots, delete the corresponding row from the table.
                Begin!

                Question: {input}
                Thought:{agent_scratchpad}'''

        emergency_info_agent = create_sql_agent(
            llm=llm,
            toolkit=SQLDatabaseToolkit(db=SQLDatabase.from_uri("sqlite:///emergency.db"), llm=llm),
            prompt=PromptTemplate.from_template(template),
            verbose=True,
            handle_parsing_errors=True,
        )
        return emergency_info_agent.invoke(query)['output']

    def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported")