import os
import json
from flask import Flask, request, make_response
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
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
import json

# Set OpenAI API key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

class Chatbot:
    def __init__(self, chat_history_file='chat_history.json'):
        self.chat_history_file = chat_history_file
        self.vectorstore = None
        self.conversation_chain = None
        self.load_or_generate_data()

    # All your existing functions go here as methods, with the 'self' parameter added
    def read_pdf(self, pdf_name):
        text = ""
        pdfFileObj = open(pdf_name,'rb')
        pdf_reader=PyPDF2.PdfReader(pdfFileObj)
        # Loop over the pages in the PDF and extract the text
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

    def get_pdfs(self):
        # Your existing code...
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

    def get_text_file_content(self, txt_files):
        text = ""
        for txt_file in txt_files:
            with open(txt_file, 'r') as file:
                text += file.read()
        return text

    def get_text_chunks(self, raw_text):
        splitter = CharacterTextSplitter(
            separator = "\n",
            chunk_size = 1000,
            chunk_overlap = 100,
            length_function = len
        )
        chunks = splitter.split_text(raw_text)
        return chunks

    def get_vectorstore(self, chunks):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts = chunks, embedding = embeddings)
        return vectorstore

    def get_conversation_chain(self, vectorstore):
        llm = ChatOpenAI(temperature=0.4)
        memory = ConversationBufferMemory(llm = llm, memory_key='chat_history', return_messages=True, output_key='answer')
        conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),memory=memory)
        return conversation_chain

    def get_text_from_url(self, url):
        # Send HTTP request to URL
        response = requests.get(url)

        # Parse HTML and save to BeautifulSoup object
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract text from the parsed HTML
        urltext = soup.get_text()

        # Return the text
        return urltext

    def save_chat_history(self, chat_history):
        with open(self.chat_history_file, 'w') as file:
            json.dump(chat_history, file)

    def load_chat_history(self):
        try:
            with open(self.chat_history_file, 'r') as file:
                file_content = file.read()
                if not file_content:
                    return []  # The file is empty, return an empty list
                try:
                    return json.loads(file_content)  # Try to parse the JSON
                except json.JSONDecodeError:
                    return []  # The file doesn't contain valid JSON, return an empty list
        except FileNotFoundError:
            return []  # The file doesn't exist, return an empty list

    def load_or_generate_data(self):
        # Get all the PDFs
        pdf_docs = self.get_pdfs()

        # Check if there are any PDFs in the ckbDocs folder
        if len(pdf_docs) == 0:
            print("No PDF files found in ckbDocs folder.")
            return

        # Read the text from the combined PDF
        raw_text = self.read_pdf("allminutes.pdf")

        # Split the raw text into chunks
        chunks = self.get_text_chunks(raw_text)
        # Generate a vector store from the chunks
        self.vectorstore = self.get_vectorstore(chunks)
        # Generate a conversation chain from the vector store
        self.conversation_chain = self.get_conversation_chain(self.vectorstore)

    def handle_message(self, message_text):
        # Load the chat history
        chat_history = self.load_chat_history()

        # Run the conversation chain with the user's query and the chat history
        response = self.conversation_chain.run({"question": message_text, "chat_history": chat_history})

        # Add the new conversation to the chat history
        chat_history.append({"question": message_text, "response": response})

        # Save the updated chat history
        self.save_chat_history(chat_history)

        # Return the response
        return response

app = Flask(__name__)

slack_token = os.environ["SLACK_BOT_TOKEN"]  # Set this environment variable with your Bot User OAuth Access Token
client = WebClient(token=slack_token)

chatbot = Chatbot()

@app.route('/slack/events', methods=['POST'])
def handle_event():
    data = request.get_json()
    if "challenge" in data:
        return make_response(data["challenge"], 200, {"content_type": "application/json"})

    # Check if the event is a message
    if "event" in data and data["event"]["type"] == "message" and 'bot_id' not in data['event']:
        # Extract the message text and channel ID
        message_text = data["event"]["text"]
        channel_id = data["event"]["channel"]

        # Get the response from the chatbot
        response = chatbot.handle_message(message_text)

        # Send the chatbot's response to the Slack channel
        try:
            response = client.chat_postMessage(
                channel=channel_id,
                text=response)
        except SlackApiError as e:
            assert e.response["ok"] is False
            assert e.response["error"]
            print(f"Got an error: {e.response['error']}")

    return make_response("", 200)

if __name__ == "__main__":
    app.run(port=3000)
