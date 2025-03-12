from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings


#extracting data from pdf 
def load_pdf_file(data):
    loader= DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

#performing the chinking operation 
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

#download embedding model to perform embedding 384 vector dimension
def download_hugging_face_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

