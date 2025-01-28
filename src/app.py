from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Task, Crew, Agent
import os
from DiagnosticInfoAgent import DiagnosticInfoAgent
from DoctorInfoAgent import DoctorInfoAgent
from EmergencyServicesAgent import EmergencyServicesAgent
from HospitalComparisonAgent import HospitalComparisonAgent
from constants import MODEL_NAME, OPENAI_API_KEY
from textwrap import dedent

# FastAPI app initialization
app = FastAPI(title="Health-Sense AI")

# Setting OpenAI API key and model name as environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_MODEL_NAME"] = MODEL_NAME

# Initialize agents
hospital_comparison_agent = HospitalComparisonAgent()
doctor_info_agent = DoctorInfoAgent()
emergency_services_agent = EmergencyServicesAgent()
diagnostic_info_agent = DiagnosticInfoAgent()

hospital_info_agent = hospital_comparison_agent.hospital_info_agent
doctor_slots_agent = doctor_info_agent.doctor_slots_agent
emergency_agent = emergency_services_agent.emergency_agent
diagnostic_info_agent = diagnostic_info_agent.diagnostic_info_agent

# Define a router agent
router_agent = Agent(
    role='Researcher',
    goal='Research and analyze information',
    backstory='Expert at gathering and analyzing information',
    allow_delegation=True
)


# Define the input model for queries
class QueryInput(BaseModel):
    query: str


@app.post("/query")
async def handle_query(input: QueryInput):
    query = input.query

    # Create the routing task based on the query
    routing_task = Task(
        description=dedent(f"""
            Route this query: "{query}"
            Choose the most appropriate agent based on keywords and context.
            Return the output of the selected agent.
        """),
        expected_output='',
        agent=router_agent
    )

    # Create a crew with all agents and the routing task
    multi_agent_crew = Crew(
        agents=[router_agent, hospital_info_agent, doctor_slots_agent, emergency_agent, diagnostic_info_agent],
        tasks=[routing_task]
    )

    # Execute the crew and handle results
    result_with_router = multi_agent_crew.kickoff()

    if not result_with_router.tasks_output:
        raise HTTPException(status_code=500, detail="No output generated by the agents.")

    # Collect and return the results
    responses = [task.raw for task in result_with_router.tasks_output]
    return {"query": query, "responses": responses}

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
