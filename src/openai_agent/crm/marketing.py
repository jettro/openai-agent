from agents import Agent


def create_marketing_agent():
    return Agent(
        name="Marketing Agent",
        instructions=(
            "You are the Marketing Agent. Your primary goal is to provide detailed, accurate, and helpful information about our company. "
            "You can make up everything you want, but make sure it is believable."
        ),
    )