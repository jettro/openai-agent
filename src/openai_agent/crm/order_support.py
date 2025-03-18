import os
import pickle

from agents import Agent, function_tool, set_default_openai_key, Runner, RunConfig, RunContextWrapper, RunResult
from dotenv import load_dotenv
from pandas import DataFrame
from pydantic import BaseModel

base_dir = os.path.dirname(os.path.abspath(__file__))
database_file_path = os.path.join(base_dir, "../../../export/database.pickl")

class Order(BaseModel):
    user_id: str
    order_id: str
    status: str
    delivery_date: str


class UserInfo(BaseModel):
    user_id: str
    user_name: str

def load_orders() -> DataFrame:
    """Load the orders from the database"""
    with open(database_file_path, 'rb') as f:
        data = pickle.load(f)

    return data


@function_tool
def find_order_information(wrapper: RunContextWrapper[UserInfo], order_id: str) -> Order | str:
    """Find the information about an order, if no order is found return a message
    telling that the order was not found.

    Args:
        order_id (str): The order id to find the information for
    """
    orders = load_orders()
    user_orders = orders.query("user_id == @wrapper.context.user_id and order_id == @order_id")

    if user_orders.empty:
        return f"Order with id {order_id} not found for user {wrapper.context.user_name}."

    order = user_orders.iloc[0]
    return Order(
        user_id=order["user_id"],
        order_id=order["order_id"],
        status=order["status"],
        delivery_date=order["delivery_date"]
    )


def init_database():
    """Initialize the database with some orders"""
    data = DataFrame({
        "user_id": ["1", "2"],
        "order_id": ["123", "124"],
        "status": ["shipped", "processing"],
        "delivery_date": ["2022-06-15", None]
    })

    with open(database_file_path, 'wb') as file_write:
        pickle.dump(data, file_write)


def create_order_support_agent():
    """Create an agent that can help with order support"""
    # Check for existing database
    if not os.path.exists(database_file_path):
        init_database()

    return Agent(
        name="Order Support",
        instructions=(
            "You are the Order Support Agent. Your primary goal is to provide detailed, accurate, and helpful information about orders. "
            "Use only the information from the FileSearchTool. If nothing is available, do not make up information, tell that you do not know."
        ),
        tools=[
            find_order_information
        ]
    )

def execute_agent(_agent: Agent, input_text: str | list, _user_info: UserInfo) -> RunResult:
    return Runner.run_sync(
        starting_agent=_agent,
        input=input_text,
        run_config=RunConfig(workflow_name="Extract professional info"),
        context=_user_info
    )


if __name__ == "__main__":
    _ = load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Make the API key available so the tracing also works
    set_default_openai_key(api_key)

    agent = create_order_support_agent()
    user_info = UserInfo(user_id="1", user_name="Jettro")
    result =  execute_agent(_agent=agent, input_text="How can I find information about my order?", _user_info=user_info)

    print(result.final_output)

    memory = result.to_input_list() + [{'role':'user', 'content':'The order id is 123.'}]
    result = execute_agent(_agent=agent, input_text=memory, _user_info=user_info)

    print(result.final_output)
