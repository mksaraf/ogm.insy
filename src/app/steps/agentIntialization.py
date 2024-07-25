from langchain.agents import initialize_agent

def intializeAgent(tools,llm2,conversational_memory2):
    agent = initialize_agent(
        agent='chat-conversational-react-description',
        tools=tools,
        llm=llm2,
        verbose=True,
        max_iterations=10,
        early_stopping_method='generate',
        memory=conversational_memory2
    )
    return agent