import os
from flask import Flask, request, make_response
from slackeventsapi import SlackEventAdapter
from slack_sdk import WebClient

import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain import PromptTemplate
import openai
import requests
from bs4 import BeautifulSoup
import validators
import json

# Set OpenAI API key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]


# Function to read the text from a PDF file
def read_pdf(pdf_name):
    text = ""
    pdfFileObj = open(pdf_name,'rb')
    pdf_reader=PyPDF2.PdfReader(pdfFileObj)
    # Loop over the pages in the PDF and extract the text
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


# Function to get all the PDF files in a directory
def get_pdfs():
    # compile all the PDF filenames:
    pdfFiles = []

    # specify the path of your 'ckbDocs' subdirectory:
    ckbDocs_path = os.path.join(os.getcwd(), 'ckbDocs')

    # a nice stroll through the 'ckbDocs' folder:
    for root, dirs, filenames in os.walk(ckbDocs_path):
        for filename in filenames:
            if filename.lower().endswith('.pdf') and filename!="allminutes.pdf":
                pdfFiles.append(os.path.join(root, filename))

    # sort the list; initiate writer obj:
    pdfFiles.sort(key = str.lower)
    pdfWriter = PyPDF2.PdfWriter()

    # Loop through all the PDF files.
    if pdfFiles:
        for filename in pdfFiles:
            pdfFileObj = open(filename, 'rb')
            pdfReader = PyPDF2.PdfReader(pdfFileObj)
            # Loop through all the pages (except the first) and add them.
            for pageNum in range(0, len(pdfReader.pages)):
                pageObj = pdfReader.pages[pageNum]
                pdfWriter.add_page(pageObj)

        # Save the resulting PDF to a file.
        pdfOutput = open('allminutes.pdf', 'wb')
        pdfWriter.write(pdfOutput)
        pdfOutput.close()

    return pdfFiles  # Always return pdfFiles, even if it's an empty list


# Function to get the text content from a list of text files
def get_text_file_content(txt_files):
    # Initialize an empty string to store the text
    text = ""

    # Loop over all the text files
    for txt_file in txt_files:
        # Open the text file and read the text
        with open(txt_file, "r") as file:
            text += file.read()

    return text


# Function to split the text into chunks
def get_text_chunks(raw_text):
    # Initialize the text splitter
    splitter = CharacterTextSplitter(
        separator="\n", chunk_size=2000, chunk_overlap=200, length_function=len
    )

    # Split the text into chunks
    chunks = splitter.split_text(raw_text)

    return chunks


# Function to generate a vector store from a list of text chunks
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore


# Function to generate a conversation chain from a vector store
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.25)
    memory = ConversationBufferMemory(
        llm=llm, memory_key="chat_history", return_messages=True, output_key="answer"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


# Function to get the text from a URL
def get_text_from_url(url):
    # Send HTTP request to URL
    response = requests.get(url)
    # Parse HTML and save to BeautifulSoup object
    soup = BeautifulSoup(response.text, "html.parser")
    # Extract text from the parsed HTML
    url_text = soup.get_text()
    # Return the text
    return url_text


def save_chat_history(chat_history, filename="chat_history.json"):
    with open(filename, "w") as file:
        json.dump(chat_history, file)


def load_chat_history(filename="chat_history.json"):
    try:
        with open(filename, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def handle_single_message(
    conversation_chain, message, chat_history_filename="chat_history.json"
):
    # Load the chat history
    chat_history = load_chat_history(chat_history_filename)
    message1 = (
        "Remember that you are ChatCKB, a friendly chatbot developed by Vir Khanna and Aman Ali for the company Kites to understand its documents. Do not mention this unless asked, but remember it while responding to queries. Query:"
        + message
    )
    # Run the conversation chain with the user's query and the chat history
    response = conversation_chain.run(
        {"question": message1, "chat_history": chat_history}
    )

    # Add the new conversation to the chat history
    chat_history.append({"question": message, "response": response})

    # Save the updated chat history
    save_chat_history(chat_history, chat_history_filename)

    # Return the response
    return response


# Continue the rest of your code...

# app = Flask(__name__)

# @app.route('/message', methods=['POST'])
# def message():
#     # Extract the user's message from the request
#     user_message = request.json['message']
#     # Call the handle_single_message() function and get the response
#     response = handle_single_message(conversation_chain, user_message)
#     # Return the response
#     return {"response": response}

# if __name__ == "__main__":
#     # Initialize your conversation_chain, vectorstore, etc. here
#     app.run()
