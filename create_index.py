import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec # Tells where our index is hosted

load_dotenv() #Call it so that variables are ready

if __name__ == "__main__":
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])  #Create Pinecone client with our API key from the environment
    index_name = os.environ["INDEX_NAME"]  #INDEX NAME comes from environment too

    if pc.has_index(index_name):  #has_index checks whether an index with this name already exists
        print(f"Index '{index_name}' found, deleting...") #If it does, print that we found it
        pc.delete_index(index_name) #Delete it so that every run starts with an empty index. 
                                    #Note : This removes everything stored in it

    pc.create_index(
        name=index_name,
        vector_type="dense", #Dense vector is a list of numbers that captures the meaning of a text 
        dimension=1536,  # OpenAIEmbeddings default (text-embedding-3-small) ; Length of the vector OpenAI's text embedding 3 small gives us by default. Index must match the embedding model we use later
        metric="cosine",  # How Pinecone measures how similar two vectors are. 
        spec=ServerlessSpec(cloud="aws", region="us-east-1"), #ServerlessSpec says where the index is hosted 
    )
    print(f"Index '{index_name}' created")

#Pinecone runs on AWS, GCP and Azure. Region decides where data lives. Region decides latency too. Keep app and data close
#Once index is created, cloud and region cant be changed