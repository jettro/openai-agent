import os

from agents import Agent, WebSearchTool, set_default_openai_key, Runner, function_tool, RunConfig
from dotenv import load_dotenv


@function_tool
def write_content_to_file(content:str, file_name: str) -> str:
    """Write the content to a file

    Args:
        content (str): The content to write to the file
        file_name (str): The name of the file to write to, always use extension 'md'
    """

    file_path = f"../../export/{file_name}"
    try:
        with open(file_path, 'w') as file:
            file.write(content)
        return "The content has been written to the file."
    except Exception as e:
        return f"Error writing to file {file_path}: type: {type(e)}, message: {str(e)}"


def main():
    _ = load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')

    # Make the API key available so the tracing also works
    set_default_openai_key(api_key)

    agent = Agent(
        name="Investigator Assistant",
        instructions=f"""
You are an expert in gathering the right information from the web about peoples activities. You goal is to find information about people that they can use for their CV. Be thorough, find at least infomation about, current position, blogs, presentations, education. Generate all information in markdown format, use references to the sources and make sure the information is correct. Use the 'write_content_to_file' tool to write the information to a file.""",
        tools=[
            WebSearchTool(),
            write_content_to_file,
        ],
    )

    return Runner.run_sync(
        starting_agent=agent,
        input="Help me find professional information about Marijn Coenradie, focus on recent information (last 5 years) and focus on search and machines learning experience.",
        run_config=RunConfig(workflow_name="Extract professional info")
    )

if __name__ == "__main__":
    result = main()
    print(result.final_output)

