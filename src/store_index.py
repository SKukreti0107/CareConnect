import os 
from helper import gemini_api_embeddings, load_pdf_file,text_split,gemini_api_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from time import sleep
from tqdm import tqdm
import math

load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")

if PINECONE_API_KEY is not None:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
else:
    raise ValueError("PINECONE_API_KEY environment variable is not set")

extracted_data=load_pdf_file()
print("Extracted data loaded")
text_chunks = text_split(extracted_data)
print("Text chunks created")
embeddings = gemini_api_embeddings()
print("Embeddings created")

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "care-connect"

BATCH_SIZE = 1000  # Process 1000 documents per batch
DELAY_BETWEEN_BATCHES = 60  # 60 seconds between batches
total_chunks = len(text_chunks)
num_batches = math.ceil(total_chunks / BATCH_SIZE)

pc.create_index(
    name=index_name,
    dimension=768,
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "aws",
            "region": "us-east-1"
        }
    }
)

print("Index created")




for i in tqdm(range(num_batches)):
    start_idx = i * BATCH_SIZE
    end_idx = min((i + 1) * BATCH_SIZE, total_chunks)
    current_batch = text_chunks[start_idx:end_idx]

    #embedding each chunk into pinecone index 
    docsearch = PineconeVectorStore.from_documents(
        documents=current_batch,
        index_name=index_name,
        embedding=embeddings
    )
    if i < num_batches - 1:  # Don't wait after the last batch
        print(f"\nWaiting {DELAY_BETWEEN_BATCHES} seconds before next batch...")
        sleep(DELAY_BETWEEN_BATCHES)

