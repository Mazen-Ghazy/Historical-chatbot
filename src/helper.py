from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_pdf(data):
    loader= DirectoryLoader(data,glob="*.pdf",loader_cls=PyPDFLoader)
    document=loader.load()
    return document


def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=800,chunk_overlap=50)
    text_chunk=text_splitter.split_documents(extracted_data)
    return text_chunk

def embedding_model():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

embedding=embedding_model()
    