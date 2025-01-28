from crewai import Task, Crew, Agent
import os
# Importing custom agents from other modules
from DiagnosticInfoAgent import DiagnosticInfoAgent
from DoctorInfoAgent import DoctorInfoAgent
from EmergencyServicesAgent import EmergencyServicesAgent
from HospitalComparisonAgent import HospitalComparisonAgent
from constants import MODEL_NAME, OPENAI_API_KEY
from textwrap import dedent

# Setting the OpenAI API key and model name as environment variables
os.environ["OPENAI_API_KEY"]= OPENAI_API_KEY
os.environ["OPENAI_MODEL_NAME"] = MODEL_NAME

# Initialize agents for hospital comparison, doctor information, emergency services,
# and diagnostic information
hospital_comparison_agent = HospitalComparisonAgent()
doctor_info_agent = DoctorInfoAgent()
emergency_services_agent = EmergencyServicesAgent()
diagnostic_info_agent = DiagnosticInfoAgent()

# Initialize the individual agents from the agent objects
hospital_info_agent = hospital_comparison_agent.hospital_info_agent
doctor_slots_agent = doctor_info_agent.doctor_slots_agent
emergency_agent = emergency_services_agent.emergency_agent
diagnostic_info_agent = diagnostic_info_agent.diagnostic_info_agent

# Create tasks that correspond to the agents
hospital_info_task = Task(description="which hospital has good medical imaging",
                        expected_output='',
                        agent=hospital_info_agent)
doctor_slots_task = Task(description="show slots for lee",
                         expected_output='',
                         agent=doctor_slots_agent)
emergency_task = Task(description="any ambulance available at 94404",
                      expected_output='',
                      agent=emergency_agent)
diagnostic_info_task = Task(description="what would be preparation instructions for cancer screening",
                        expected_output='',
                        agent=diagnostic_info_agent)

# Define a router agent that decides which task to assign based on the query
router_agent = Agent(
    role='Researcher',
    goal='Research and analyze information',
    backstory='Expert at gathering and analyzing information',
    allow_delegation=True
)

# Example query that the router agent will route to the appropriate agent
query = "show slots for dr lee"

# Create the routing task, which will determine which agent should handle the query
routing_task = Task(
        description=dedent(f"""
            Route this query: "{query}"
            Choose the most appropriate agent based on keywords and context.
            Return the output of the selected agent.
        """),
        expected_output='',
        agent=router_agent
)

# Create a crew with all agents and tasks, which will be executed in parallel
multi_agent_crew = Crew(
    agents=[router_agent, hospital_info_agent, doctor_slots_agent, emergency_agent, diagnostic_info_agent],
    tasks=[routing_task]
)

# Execute the crew and get the results
result_with_router = multi_agent_crew.kickoff()

# Print the output of each task in the result
for task in result_with_router.tasks_output:
    print(task.raw)
    print("\n\n")
