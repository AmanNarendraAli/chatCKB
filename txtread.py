import os
from PyPDF2 import PdfReader 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch,Pinecone,Weaviate,FAISS

os.environ["OPENAI_API_KEY"] = "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"

def get_text_file_content(txt_files):
    text = ""
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            text += file.read()
    return text

def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = splitter.split(raw_text)
    return chunks

txt_files = ["testdoc.txt"]
raw_text = get_text_file_content(txt_files)

print(raw_text)
