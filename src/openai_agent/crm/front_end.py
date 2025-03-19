import os

from agents import Agent, set_default_openai_key, Runner, RunConfig, RunResult
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from dotenv import load_dotenv

from openai_agent.crm.marketing import create_marketing_agent
from openai_agent.crm.order_support import create_order_support_agent, UserInfo
from openai_agent.crm.product_expert import create_product_expert_agent


def create_front_end_agent():
    vector_store_id = os.getenv('VECTOR_STORE_ID')
    product_expert_agent = create_product_expert_agent(vector_store_id)
    order_support_agent = create_order_support_agent()
    marketing_agent = create_marketing_agent()

    return Agent(
        name="Front End Agent",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}"
            "\nThe front-desk agent receives the incoming request. Its goal is to learn the desires of the caller. "
            "Is it contacting to discuss a problem with an order. Or is it contacting you with a question about a "
            "product? Maybe the caller has a question about the company. The case is handed over to a "
            "second-line support agent if the front-desk agent has enough information. "
        ),
        handoffs=[
            product_expert_agent, order_support_agent, marketing_agent
        ]
    )


def execute_agent(_agent: Agent, input_text: str | list, _user_info: UserInfo) -> RunResult:
    if isinstance(input_text, list):
        print(f"Agent '{_agent.name}' receives additional information '{input_text[-1]["content"]}'")
    else:
        print(f"Agent '{_agent.name}' receives the question '{input_text}'")

    return Runner.run_sync(
        starting_agent=_agent,
        input=input_text,
        run_config=RunConfig(workflow_name="Full Support Flow"),
        context=_user_info
    )



if __name__ == "__main__":
    _ = load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Make the API key available so the tracing also works
    set_default_openai_key(api_key)

    front_end_agent = create_front_end_agent()

    user_info = UserInfo(user_id="1", user_name="Jettro")
    result  = execute_agent(_agent=front_end_agent, input_text="I have a question about my order", _user_info=user_info)
    print(f"Response from agent '{result.last_agent.name}'\n{result.final_output}")

    messages = result.to_input_list()
    messages = messages + [{'role': 'user', 'content': 'My order ID is 123.'}]
    result = execute_agent(_agent=result.last_agent, input_text=messages, _user_info=user_info)

    print(f"Response from agent '{result.last_agent.name}'\n{result.final_output}")

    # result = execute_agent(_agent=front_end_agent, input_text="What can you tell me about the vision of the company?", _user_info=user_info)
    # print(f"Talking to agent '{result.last_agent.name}'")
    # print(result.final_output)