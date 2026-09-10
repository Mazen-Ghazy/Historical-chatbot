from langchain_chroma import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

loader= DirectoryLoader(data="data",glob="*.pdf",loader_cls=PyPDFLoader)
document=loader.load()

text_splitter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=50)
text_chunk=text_splitter.split_documents(document)

embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

verctor_db=Chroma.from_documents(persist_directory="vectordb",embedding_function=embedding)
