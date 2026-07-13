# test_run.py
from agent_framework.core.tasks import Task
from agent_framework.agents.mock import MockAgent
from agent_framework.orchestration.registry import AgentRegistry

# Create the registry
registry = AgentRegistry()

# Register our Mock Agent
registry.register("mock_agent", MockAgent)

# Look up the agent in the phonebook and create it
AgentClass = registry.get("mock_agent")
my_agent = AgentClass(name="Testy", role="Tester", system_prompt="Just testing")

# Create a Task card
task = Task(description="Write a hello world program", assigned_agent="Testy")

# Make the agent execute the task
print("Starting Test")
result = my_agent.execute(task)

# Print out the results to make sure everything connected perfectly
print(f"Task ID: {result.task_id}")
print(f"Success Status: {result.success}")
print(f"Agent Output: {result.output}")