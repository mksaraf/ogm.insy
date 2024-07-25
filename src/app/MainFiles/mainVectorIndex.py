"""
Create the chatbot OGM.Insy for life insurance

Author: Geetika Saraf & Manish Kumar Saraf

"""
# Import library

from steps.key import getOpenAiKey, getPineconeKey
from steps.dataSource import checkDataSource, dataLoad
from steps.chunking import chunking
from steps.Index import CheckAndCreateIndex
from steps.utils import ConnectToPinecone
from steps.uploadVectorsToIndex import UploadVectorsToIndex
from steps.embeddings import embedding

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as pine


import warnings
warnings.filterwarnings('ignore')

# Arguments
data_path = "/Users/geetikasaraf/Library/Mobile Documents/com~apple~CloudDocs/OGM/OGM.Data/North America"
chunk_size=1000
chunk_overlap=0
index_name='life-insurance-index1'

def main():

    # Step 1: Get OpenAI Key
    openAiKey=getOpenAiKey()
    print("Step 1 Get Open AI Key:", openAiKey)

    # Step 2: Connect with Data Source
    dataExist = checkDataSource(data_path)
    print("Step 2 Data Source:", dataExist)

    # Step 3: Load data source
    data = dataLoad(data_path)
    print("Step 3 Load Data", data[0])
    # Chunking
    chunks= chunking(data,chunk_size,chunk_overlap)
    print("Step 4 Length of chunks:", len(chunks))

    # Connect to Vector-DB Pinecone
    pc=ConnectToPinecone()
    print("Step 5 Vector-DB:", CheckAndCreateIndex(index_name,pc))

    # Upload vectors into Pinecone vector DB
    index = pc.Index(index_name)
    embeddings = embedding()
    UploadVectorsToIndex(chunks,embeddings,index)
    print("Step 6 Describe Index stats:", index.describe_index_stats())

    


    # Fetch from Vector-DB


    # Que and Answer

    # Memory

    # Agent

    # Evaluation

    # Moderation

    # Deployment

if __name__ == '__main__':
    main()
