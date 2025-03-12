from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")

if PINECONE_API_KEY is not None:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
else:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

extracted_data = load_pdf_file(data='Data/')
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "care-connect"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    }
)

#embedding each chunk into pinecone index 
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

