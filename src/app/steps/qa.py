from langchain.chains import RetrievalQA


# Create QA
def qa(chainType,vectorstore2,llm2):
    qa2 = RetrievalQA.from_chain_type(
        llm=llm2,
        chain_type=chainType,
        retriever=vectorstore2.as_retriever()
    )
    return qa2

