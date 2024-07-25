
import numpy as np
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
import streamlit as st
from steps.utils import ConnectToPinecone
from steps.key import getPineconeKey
from steps.key import getOpenAiKey, getPineconeKey
from steps.dataSource import checkDataSource, dataLoad
from steps.chunking import chunking
from steps.Index import CheckAndCreateIndex
from steps.key import getPineconeKey
from steps.uploadVectorsToIndex import UploadVectorsToIndex
from steps.utils import ConnectToPinecone
from steps.llm import llmModel
from steps.qa import qa
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from steps.embeddings import embedding
from steps.createVectorstore import CallVectorStore
from steps.memory import addMemory


import warnings
warnings.filterwarnings('ignore')

# Arguments
data_path = "/Users/geetikasaraf/Library/Mobile Documents/com~apple~CloudDocs/OGM/OGM.Data/North America"
chunk_size=1000
chunk_overlap=0
openAiKey=getOpenAiKey()
embeddings=embedding()
text_field = "text"
model='gpt-3.5-turbo'
temperature=0.0
chainType="stuff"
index_name='life-insurance-index1'

pc=ConnectToPinecone()
print("Step 1 Vector-DB:", CheckAndCreateIndex(index_name,pc))

# Find Pinecone index and connect
index = pc.Index(index_name)

print("Step 2 Describe Index stats:", index.describe_index_stats())
    
    # Create LLM
llm2 = llmModel(model, temperature, openAiKey)
   # # Create LLM
vectorstore2 = CallVectorStore(index)


    # Memory
memory=addMemory()


# # Pinecone Cloud settings
# pinecone_api_key = "a2a08fb6-dbff-459d-9f99-dc48b0b7cbdd"
# pinecone_api_key=getPineconeKey()
# # pinecone_environment = "us-west2"
# vector_db_name = "life-insurance-index1"

# # Load the Pinecone Cloud vector database
# #pinecone_client = pinecone.Client(api_key=pinecone_api_key, environment=pinecone_environment)
# pinecone_client = ConnectToPinecone()
# vector_db = pinecone_client.vector_database(vector_db_name)

# pinecone_client=pinecone(pinecone_api_key)
# vector_db=pinecone_client.Index(vector_db_name)

# # Load the pre-trained language model and tokenizer
# model_name = "distilbert-base-uncased-distilled-squad"
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(llm2)

# # Define the agent's memory capacity (adjust as needed)
memory_capacity = 1000

# # Create a memory buffer to store previous conversations
memory_buffer = []

def process_input(input_text):
    # Tokenize the input text
    inputs = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        max_length=512,
        return_attention_mask=True,
        return_tensors='pt'
    )

    # Get the input vector from the Pinecone Cloud vector database
    #input_vector = vector_db.get_vectors(inputs['input_ids'].numpy().reshape(1, -1))[0]

#     # Compute the similarity between the input vector and the memory buffer
#     similarities = np.dot(memory_buffer, input_vector)

#     # Find the most similar memory entry
#     idx = np.argmax(similarities)
#     memory_entry = memory_buffer[idx]

#     # Generate a response based on the most similar memory entry
#     response = generate_response(memory_entry, input_text)

#     # Update the memory buffer
#     update_memory_buffer(input_text, response)

#     return response

def generate_response(memory_entry, input_text):
    # Use a simple template-based response generator for now
    # You can replace this with a more advanced NLP model or technique
    response_template = "I remember you asked me about {} before. Is that related to what you're asking now?"
    response = response_template.format(memory_entry[1])
    return response

def update_memory_buffer(input_text, response):
    global memory_buffer
    memory_buffer.append((input_text, response))
    if len(memory_buffer) > memory_capacity:
        memory_buffer.pop(0)  # Remove the oldest memory entry

def main():
    st.title("Conversational Agent")
    st.header("Chat with me!")

    input_text = st.text_input("You: ")

    if input_text:
        response = process_input(input_text)
        st.text_area("Agent: ", value=response, height=100)

if __name__ == "__main__":
    main()