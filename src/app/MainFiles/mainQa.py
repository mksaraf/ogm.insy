"""
Create the chatbot OGM.Insy for life insurance

Author: Geetika Saraf & Manish Kumar Saraf

"""
# Import library

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


def main():

    # Connect to Vector-DB Pinecone
    pc=ConnectToPinecone()
    print("Step 1 Vector-DB:", CheckAndCreateIndex(index_name,pc))

    # Find Pinecone index and connect
    index = pc.Index(index_name)

    print("Step 2 Describe Index stats:", index.describe_index_stats())
    
    # Create LLM
    llm2 = llmModel(model, temperature, openAiKey)

    # Call VectorStore using index
    vectorstore2 = CallVectorStore(index)

    # Create QA chain
    quesAns = qa(chainType,vectorstore2,llm2)

    #Testing QA chain
    # query=input("Hi I am Insy. Ask me Question related to life insurance? \n")
    # output=quesAns.invoke(query)
    # print(output)

    while True:
    
        user_input=input("Hi I am Insy. Ask me Que? \n")
        
        if user_input.lower()=='quit':
            break
        else:
            answer=quesAns.invoke(user_input)
            print(answer)
 

if __name__ == '__main__':
    main()
