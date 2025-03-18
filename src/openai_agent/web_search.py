import asyncio

from agents import Agent, FileSearchTool, Runner, WebSearchTool

agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
    ],
)

async def main():
    result = await Runner.run(agent, "What is the profession of Jettro Coenradie?")
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())