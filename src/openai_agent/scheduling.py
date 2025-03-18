import os

from agents import Agent, WebSearchTool, set_default_openai_key, Runner, function_tool, RunConfig
from dotenv import load_dotenv

@function_tool
def find_availability_for(name: str) -> str:
    """Find the availability of a person

    Args:
        name (str): The name of the person to find the availability for
    """
    if name == "Alice":
        return f"{name} is available on monday morning and wednesday afternoon."
    if name == "Bob":
        return f"{name} is available on monday afternoon and wednesday."
    return f"I do not have access to the schedule of that person."


def main():
    _ = load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Make the API key available so the tracing also works
    set_default_openai_key(api_key)

    scheduling_agent = Agent(
        name="Scheduling Assistant",
        instructions=f"""You assist people in making appointments. You have access to the schedule of people and find a common available time slot.""",
        tools=[
            find_availability_for,
        ]
    )

    personal_assistant = Agent(
        name="personal assistant",
        instructions=f"""You assist people in making appointments. You have access to other agents that can help you with this task. 
""",
        tools=[
            scheduling_agent.as_tool(
                tool_name="schedule_appointment",
                tool_description="Check the common availability of persons.",
            )
        ],
    )

    return Runner.run_sync(
        starting_agent=personal_assistant,
        input="I like to get book an appointment between Bob and Alice.",
        run_config=RunConfig(workflow_name="Coordinate appointment")
    )

if __name__ == "__main__":
    result = main()
    print(result.final_output)

