
from steps.embeddings import embedding
from langchain.vectorstores import Pinecone as pine

 #Create vectorstore
    
embeddings=embedding()
def CallVectorStore(index):
    text_field = "text"
    vectorstore = pine(index, embeddings.embed_query, text_field)
    return vectorstore