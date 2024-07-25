from langchain.chat_models import ChatOpenAI
from steps.key import getOpenAiKey



# Create llm
def llmModel(model, temperature,openAiKey):
    llm = ChatOpenAI(
        openai_api_key=openAiKey,
        model_name=model,
        temperature=temperature
    )
    return llm







