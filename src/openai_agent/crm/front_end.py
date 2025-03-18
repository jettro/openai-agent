import os

from agents import Agent, set_default_openai_key, Runner, RunConfig
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

if __name__ == "__main__":
    _ = load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Make the API key available so the tracing also works
    set_default_openai_key(api_key)

    front_end_agent = create_front_end_agent()

    user_info = UserInfo(user_id="1", user_name="Jettro")
    result  = Runner.run_sync(
        starting_agent=front_end_agent,
        input="I like to obtain information about my order with id 123.",
        run_config=RunConfig(workflow_name="Contact front desk"),
        context=user_info
    )

    print(result.final_output)
