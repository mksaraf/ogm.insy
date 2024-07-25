# from langchain import ConversationalRetrievalChain
from langchain.chains.conversational_retrieval import ConversationalRetrievalChain


# Create CR-QA
def crChain(memory,retriever,llm2):
    crc = ConversationalRetrievalChain.from_llm(
        memory=memory,
        retriever=retriever,
        llm=llm2
    )
    return crc
