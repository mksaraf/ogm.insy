
from steps.key import getPineconeKey
from pinecone import Pinecone

# Connect to Vector-DB Pinecone
def ConnectToPinecone():
    pinecone_api_key=getPineconeKey()
    pc=Pinecone(api_key=pinecone_api_key)
    return pc