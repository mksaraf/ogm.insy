from langchain.agents import Tool


def agentTools(qa2):
    tools = [
        Tool(
            name='Knowledge Base',
            func=qa2.invoke,
            description=(
                'use this tool when answering general knowledge queries to get '
                'more information about the topic'
            )
        )
    ]
    return tools