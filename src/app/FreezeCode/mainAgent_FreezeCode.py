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
from steps.memory import conversationalMemory
# from steps.crc import crChain
# from steps.retrievalAugQAChain import raQaChain
from steps.prompt import prompt
from steps.agentTools import agentTools
from steps.agentIntialization import intializeAgent
from langchain_core.chat_history import BaseChatMessageHistory
import warnings
warnings.filterwarnings("ignore")


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

    
    # using agent
    pc=ConnectToPinecone()
    index = pc.Index(index_name)
    vectorstore2 = CallVectorStore(index)
    llm2 = llmModel(model, temperature, openAiKey)
    qa2 = qa(chainType,vectorstore2,llm2)

    tools=agentTools(qa2)
    memory=conversationalMemory()
    agent=intializeAgent(tools,llm2,memory)
    

    while True:
    
        message=input("User:")
        
        if message.lower()=='quit':
            break
        else:
            result=agent(message)
            print(result)


    # Evaluation

    # Moderation

    # Deployment

if __name__ == '__main__':
    main()
