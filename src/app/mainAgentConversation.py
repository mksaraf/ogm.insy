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
from steps.memory import addMemory
from steps.crc import crChain
from steps.retrievalAugQAChain import raQaChain
from steps.prompt import prompt

# import warnings
# warnings.filterwarnings('ignore')

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
    print("Step 5 Vector-DB:", CheckAndCreateIndex(index_name,pc))

    # Upload vectors into Pinecone vector DB
    index = pc.Index(index_name)

    print("Step 6 Describe Index stats:", index.describe_index_stats())
    
    # # Create LLM
    llm2 = llmModel(model, temperature, openAiKey)
    vectorstore2 = CallVectorStore(index)
    quesAns = qa(chainType,vectorstore2,llm2)

    # Que and Answer
    # query=input("Hi I am Insy. Ask me Que? \n")
    # output=quesAns.invoke(query)
    # print(output)
   
    # Memory
    memory=addMemory()

    # CONVERSATIONAL RETIEVAL CHAIN
    crc=crChain(memory,vectorstore2,llm2)

    # Conversational chain question answer
    question=input("Hi I am Insy. Ask me Que? \n")
    # result = crc({"question": question})
    # print(result['answer'])

    #Retrieval Augmented QA Chain
    prompt2=prompt()
    retriever=vectorstore2.as_retriever()
    raQaChain2=raQaChain(question,retriever,llm2,prompt2)
    result=raQaChain2.invoke({"question": question})
    print(result)

    # Evaluation

    # Moderation

    # Deployment

if __name__ == '__main__':
    main()
