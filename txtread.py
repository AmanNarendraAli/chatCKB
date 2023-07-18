import os
import PyPDF2
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch,Pinecone,Weaviate,FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain import PromptTemplate
import openai
import requests
from bs4 import BeautifulSoup
import validators

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

    # flashy display:
    #print('{}\n'.format(len(pdfFiles)))
    #for i in pdfFiles:
        #print(i)

    # Loop through all the PDF files.
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

# Function to get the text content from a list of text files
def get_text_file_content(txt_files):
    text = ""
    for txt_file in txt_files:
        with open(txt_file, 'r') as file:
            text += file.read()
    return text

# Function to split the text into chunks
def get_text_chunks(raw_text):
    splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = splitter.split_text(raw_text)
    return chunks

# Function to generate a vector store from a list of text chunks
def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts = chunks, embedding = embeddings)
    return vectorstore

# Function to generate a conversation chain from a vector store
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(temperature=0.6)
    memory = ConversationBufferMemory(llm = llm, memory_key='chat_history', return_messages=True, output_key='answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),memory=memory)
    return conversation_chain

# Function to get the text from a URL
def get_text_from_url(url):
    # Send HTTP request to URL
    response = requests.get(url)

    # Parse HTML and save to BeautifulSoup object
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract text from the parsed HTML
    urltext = soup.get_text()

    # Return the text
    return urltext

# Main function to run the application
def main():
    # Get all the PDFs
    pdf_docs = get_pdfs()
    # Read the text from the combined PDF
    raw_text = read_pdf("allminutes.pdf")
    # Get the user's input
    input_string = input('Enter your question: ')

    # Check if input string is a URL
    if validators.url(input_string):
        # If it's a URL, extract the text from the webpage
        raw_text += get_text_from_url(input_string)
        query = input("Actually enter your question: ")
    else:
        # If it's not a URL, consider it as the query
        query = input_string

    # Split the raw text into chunks
    chunks = get_text_chunks(raw_text)
    # Generate a vector store from the chunks
    vectorstore = get_vectorstore(chunks)
    # Generate a conversation chain from the vector store
    conversation_chain = get_conversation_chain(vectorstore)
    chat_history = []  # Add this line if there's no chat history yet.
    # Run the conversation chain with the user's query and the chat history
    response = conversation_chain.run({"question":query,"chat_history":chat_history})
    print("gittest")
    # Print the response
    print(response)

# Function to test the LangChain chat functionality
def LangChainTest():
    chat = ChatOpenAI(temperature=1)
    prompt_template = "Tell me a {adjective} joke"
    llm_chain = LLMChain(llm=chat, prompt=PromptTemplate.from_template(prompt_template))

    print(llm_chain.run({"adjective": "corny"}))

# Run the main function
main()
