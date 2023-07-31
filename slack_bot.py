from flask import Flask, request, make_response
from slack_sdk.signature import SignatureVerifier
from slack_sdk.web import WebClient
from slack_sdk.errors import SlackSignatureVerificationError
import os
import json
import validators
import requests
from bs4 import BeautifulSoup
from main import (
    handle_single_message,
    get_conversation_chain,
    get_vectorstore,
    get_text_chunks,
    read_pdf,
    get_pdfs,
    get_text_from_url,
    load_chat_history,
    save_chat_history,
)

# Get the slack token and signing secret from the environment
slack_token = os.getenv("SLACK_BOT_TOKEN")
slack_signing_secret = os.getenv("SLACK_SIGNING_SECRET")

# Create a new WebClient - you'll use this to send messages back to Slack
slack_web_client = WebClient(token=slack_token)

# Create a new SignatureVerifier - you'll use this to verify requests from Slack
signature_verifier = SignatureVerifier(slack_signing_secret)

# Initialize a Flask app
app = Flask(__name__)

# Global variable to store the conversation chain
conversation_chain = None
chat_history_filename = "chat_history.json"  # The filename for storing chat history


@app.route("/slack/message_actions", methods=["POST"])
def message_actions():
    # Verify the request
    if not signature_verifier.is_valid_request(
        request.get_data().decode("utf-8"), request.headers
    ):
        return make_response("invalid request", 403)

    # Extract the user's message
    form_json = json.loads(request.form["payload"])
    user_message = form_json["message"]["blocks"][0]["elements"][0]["elements"][0][
        "text"
    ]

    global conversation_chain
    if conversation_chain is None:
        # Get all the PDFs
        pdf_docs = get_pdfs()

        # Check if there are any PDFs in the ckbDocs folder
        if len(pdf_docs) == 0:
            print(
                "No PDF files found in ckbDocs folder. Please enter a URL to proceed."
            )
            url = input("Enter the URL: ")

            # Check if input string is a valid URL
            if validators.url(url):
                # If it's a URL, extract the text from the webpage
                raw_text = get_text_from_url(url)
            else:
                print("Invalid URL. Please check your input.")
                return
        else:
            # Read the text from the combined PDF
            raw_text = read_pdf("allminutes.pdf")

        # Split the raw text into chunks
        chunks = get_text_chunks(raw_text)
        # Generate a vector store from the chunks
        vectorstore = get_vectorstore(chunks)
        # Generate a conversation chain from the vector store
        conversation_chain = get_conversation_chain(vectorstore)
        print("Initialized conversation chain")  # Debug print statement

    # Get the response from our bot
    bot_response = handle_single_message(
        conversation_chain, user_message, chat_history_filename
    )

    # Send the response back to Slack
    slack_web_client.chat_postMessage(
        channel=form_json["channel"]["id"], text=bot_response
    )

    return make_response("", 200)


if __name__ == "__main__":
    app.run(port=3000)
