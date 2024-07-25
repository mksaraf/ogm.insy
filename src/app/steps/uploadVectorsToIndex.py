
def UploadVectorsToIndex(chunks,embeddings,index):

    for i, t in zip(range(len(chunks)), chunks):
        query_result = embeddings.embed_query(t.page_content)
        index.upsert(
        vectors=[
                {
                    "id": str(i),  # Convert i to a string
                    "values": query_result,
                    "metadata": {"texts":str(chunks[i].page_content)} # meta data as dic
                }
            ],
            namespace="real"
)
       
     