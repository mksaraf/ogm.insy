"""
Create the chatbot OGM.Insy for life insurance

Author: Geetika Saraf & Manish Kumar Saraf

"""
# Import library

import langchain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from steps.key import getOpenAiKey, getPineconeKey
from steps.dataSource import checkDataSource, dataLoad
from steps.chunking import chunking
from steps.Index import CheckAndCreateIndex
from steps.utils import ConnectToPinecone
from steps.uploadVectorsToIndex import UploadVectorsToIndex
from steps.embeddings import embedding

from langchain_community.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as pine
from evaluation.createEvaluationDataset import CreateEvalDataset
from steps.llm import llmModel
from steps.retrievalAugQAChain import raQaChain
from evaluation.evaluationWithRAGA import create_ragas_dataset,evaluate_ragas_dataset
from steps.prompt import prompt
from steps.createVectorstore import CallVectorStore
from steps.prompt import prompt
from datasets import load_metric
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from steps.createQAChain import create_qa_chain
import warnings
warnings.filterwarnings('ignore')

# Arguments
data_path = "/Users/geetikasaraf/Library/Mobile Documents/com~apple~CloudDocs/OGM/OGM.Data/North America"
chunk_size=1000
chunk_overlap=0
question_generation_model='gpt-3.5-turbo-16k'
answer_generation_model='gpt-4-1106-preview'
rag_model='gpt-3.5-turbo'
temperature=0.0
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

    # Connect to Pinecone
    pc=ConnectToPinecone()
    embeddings=embedding()
    index = pc.Index(index_name)

    # Call pinecone vectorstore
    vectorstore2 = CallVectorStore(index)

    # Create llms
    question_generation_llm = llmModel(question_generation_model, temperature, openAiKey)
    answer_generation_llm=llmModel(answer_generation_model, temperature, openAiKey)
    
    # create retriever
    retriever=vectorstore2.as_retriever()

    # create prompt
    prompt2=prompt()

    # Passing llm for RAG
    rag_llm=llmModel(rag_model, temperature,openAiKey)

    # Create RAG pipeline
    rag_pipeline=raQaChain(retriever,rag_llm,prompt2)
    
    # Create a ground truth dataset for evaluation (Question, Context, Ground truth)
    eval_dataset=CreateEvalDataset(chunks, question_generation_llm,answer_generation_llm, openAiKey)
    #print("eval_dataset: ", eval_dataset)
    eval_dataset.to_csv("/Users/geetikasaraf/Library/Mobile Documents/com~apple~CloudDocs/OGM/OGM.Thesis/OGM.Thesis.Geetika/OGM.insy/src/app/data/evaluation_dataset_gs.csv")
  
    # Creating ragas dataset using evaluation dataset (Added answer)
    basic_qa_ragas_dataset=create_ragas_dataset(rag_pipeline, eval_dataset)
    basic_qa_ragas_dataset.to_csv("/Users/geetikasaraf/Library/Mobile Documents/com~apple~CloudDocs/OGM/OGM.Thesis/OGM.Thesis.Geetika/OGM.insy/src/app/data/raga_dataset_gs.csv")
    #print(basic_qa_ragas_dataset)
    
    # Evaluation Matrix results
    metrics_result = evaluate_ragas_dataset(basic_qa_ragas_dataset)
    metrics_result_df=metrics_result.to_pandas()
    metrics_result_df.to_csv("/Users/geetikasaraf/Library/Mobile Documents/com~apple~CloudDocs/OGM/OGM.Thesis/OGM.Thesis.Geetika/OGM.insy/src/app/data/ragas_metrics_result_gs.csv")


if __name__ == '__main__':
    main()
