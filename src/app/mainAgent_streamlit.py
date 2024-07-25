# """
# Create the chatbot OGM.Insy for life insurance

# Author: Geetika Saraf & Manish Kumar Saraf

# """
# # Import library

from steps.key import getOpenAiKey, getPineconeKey
# from steps.dataSource import checkDataSource, dataLoad
# from steps.chunking import chunking
# from steps.Index import CheckAndCreateIndex
# from steps.key import getPineconeKey
# from steps.uploadVectorsToIndex import UploadVectorsToIndex
from steps.utils import ConnectToPinecone
from steps.llm import llmModel
from steps.qa import qa
from langchain.chat_models import ChatOpenAI
# from langchain.chains import RetrievalQA
# from steps.embeddings import embedding
from steps.createVectorstore import CallVectorStore
from steps.memory import conversationalMemory
# # from steps.crc import crChain
# # from steps.retrievalAugQAChain import raQaChain
# from steps.prompt import prompt
from steps.agentTools import agentTools
from steps.agentIntialization import intializeAgent
# from langchain_core.chat_history import BaseChatMessageHistory
import warnings
warnings.filterwarnings("ignore")


# # Arguments

openAiKey=getOpenAiKey()

model='gpt-3.5-turbo'
temperature=0.0
chainType="stuff"
index_name='life-insurance-index1'




import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

import pinecone

def agent(search_results, prompt):
    # Use the search results to generate the response
    response = ""
    for result in search_results:
        response += result.page_content + "\n"
    return response

pc=ConnectToPinecone()
index = pc.Index(index_name)
vectorstore2 = CallVectorStore(index)

# Set up Langchain models
openai_api_key = getOpenAiKey()
llm2 = llmModel(model, temperature, openAiKey)
qa2 = qa(chainType,vectorstore2,llm2)

tools=agentTools(qa2)
memory=conversationalMemory()
agent=intializeAgent(tools,llm2,memory)

# Set up Streamlit app
st.title("OGM.Insy")
st.markdown("Hi I am Insy! How can I help you")

# Create a session state to store chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": "Hello!  How can I assist you today regarding financial questions?"}]

# Create a text input for user input
user_input = st.text_input("Enter your question or message:")

# Create a button to trigger the chatbot response
if st.button("Send"):
    # Get the user input and process it
    user_input_text = user_input.strip()
    if user_input_text:
        # Add the user input to the chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input_text})
        search_results = vectorstore2.similarity_search(user_input_text, k=5)
        pdf_extract = "\n ".join([result.page_content for result in search_results])
        # Get the chatbot response from Langchain
        # response = qa2(user_input_text, st.session_state.chat_history)
        # Create a prompt template and format it with the pdf extract
        prompt_template = "You asked about {}, and I found some relevant information: {}"
        prompt = [{"role": "system", "content": prompt_template.format(user_input_text, pdf_extract)}]
        prompt.append({"role": "user", "content": user_input_text})
        
        # response=agent(user_input_text, st.session_state.chat_history)

        # Get the chatbot response from Langchain, using the vector DB knowledge only
        response = agent(search_results, prompt)

        # Add the chatbot response to the chat history
        # st.session_state.chat_history.append({"role": "assistant", "content": response})

        # Add the chatbot response to the prompt and chat history
        prompt.append({"role": "assistant", "content": response})
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        # Clear the user input text
        user_input_text="" 

# Display the chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.write(f"**You:** {message['content']}")
    elif message["role"] == "assistant":
        st.write(f"**Assistant:** {message['content']}")
        

# Run the app
if __name__ == "__main__":
    # st.run_app()
    st.write("App is running")


