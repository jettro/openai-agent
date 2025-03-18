import os

from agents import Agent, Runner, set_default_openai_key, RunConfig, WebSearchTool, RunResult, FileSearchTool
from dotenv import load_dotenv


def create_product_expert_agent(_vector_store_id: str) -> Agent:
    return Agent(
        name="Product Expert",
        instructions=(
            "You are the Product Expert Agent. Your primary goal is to provide detailed, accurate, and helpful information about products. "
            "Use only the information from the FileSearchTool. If nothing is available, do not make up information, tell that you do not know."
        ),
        tools=[
            # WebSearchTool(),
            FileSearchTool(
                max_num_results=1,
                vector_store_ids=[_vector_store_id],
            ),
        ]
    )


def execute_agent(_agent: Agent, input_text: str | list) -> RunResult:
    return Runner.run_sync(
        starting_agent=_agent,
        input=input_text,
        run_config=RunConfig(workflow_name="Extract professional info")
    )


if __name__ == "__main__":
    _ = load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Make the API key available so the tracing also works
    set_default_openai_key(api_key)

    # agent = create_product_expert_agent()
    # result = execute_agent(agent=agent, input_text="Search the web for  a good transistor radio for listening to FM stations or DAB.")
    # print(result.final_output)
    #
    # memory = result.to_input_list() + [{'role':'user', 'content':'I prefer one from philips.'}]
    # result = execute_agent(agent=agent, input_text=memory)
    # print(result.final_output)

    vector_store_id = os.getenv('VECTOR_STORE_ID')
    agent = create_product_expert_agent(vector_store_id)
    # result = execute_agent(agent=agent, input_text="Do you have a headphone with that softens noice from the outside?")
    result = execute_agent(agent=agent, input_text="I want to get into shape, do you have something that can help me?")
    print(result.final_output)
