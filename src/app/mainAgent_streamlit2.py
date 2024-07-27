from steps.key import getOpenAiKey, getPineconeKey
from pinecone import Pinecone
from steps.llm import llmModel
from steps.qa import qa
from langchain.chat_models import ChatOpenAI
from steps.createVectorstore import CallVectorStore
from steps.memory import conversationalMemory
from steps.agentTools import agentTools
from steps.agentIntialization import intializeAgent
from steps.prompt import prompt
import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from openai import OpenAI
import pinecone
import warnings
warnings.filterwarnings("ignore")


# Argument
model='gpt-3.5-turbo'
temperature=0.0
chainType="stuff"
index_name='life-insurance-index1'
OPENAI_API_KEY=st.secrets['OPENAI_API_KEY']
PINECONE_API_KEY=st.secrets['PINECONE_API_KEY']

# Set up Pinecone vector dB
pc=Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(index_name)
vectorstore2 = CallVectorStore(index)
Client=OpenAI(api_key=OPENAI_API_KEY)

# Set up Langchain models
llm2 = llmModel(model, temperature, OPENAI_API_KEY)
qa2 = qa(chainType,vectorstore2,llm2)

tools=agentTools(qa2)
memory=conversationalMemory()
agent=intializeAgent(tools,llm2,memory)

st.title("INSY: A Life Insurance Virtual Agent")
st.write(
    """
This application was developed by Geetika Saraf. Insy offers a chat experience based on a knowledge base, allowing you to ask questions and receive relevant information regarding Life Insurance.

"""
)


st.info(
    """
"""
)

st.session_state["vectordb"] = vectorstore2

prompt_template = """
    You are financial expert that combines your knowledge about life insurance and data in vector db.
"""

prompt = st.session_state.get("prompt", [{"role": "system", "content": "none"}])

for message in prompt:
    if message["role"]!= "system":
           with st.chat_message(message["role"]):
                st.write(message["content"])

question = st.chat_input("Hi Insy is here to assist you!")
if question:
    vectordb = st.session_state.get("vectordb", None)
    if not vectordb:
        st.error("You need to provide a PDF")
        st.stop()

    search_results = vectordb.similarity_search(question, k=5)
    pdf_extract = "\n ".join([result.page_content for result in search_results])

    prompt[0] = {
        "role": "system",
        "content": prompt_template.format(pdf_extract=pdf_extract),
    }

    prompt.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        botmsg = st.empty()  

    response = []
    result = ""
    for chunk in Client.chat.completions.create(
        model="gpt-3.5-turbo", messages=prompt, stream=True
    ):
        text = chunk.choices[0].delta.content
        if text is not None:
            response.append(text)
            result = "".join(response).strip()

            botmsg.write(result)

    prompt.append({"role": "assistant", "content": result})

    st.session_state["prompt"] = prompt

if __name__ == "__main__":
    st.write("App is running")




