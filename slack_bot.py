from flask import Flask, request, make_response
from slackeventsapi import SlackEventAdapter
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from config import SLACK_BOT_TOKEN, SLACK_SIGNING_SECRET
from main import handle_single_message

app = Flask(__name__)

slack_event_adapter = SlackEventAdapter(SLACK_SIGNING_SECRET, "/slack/events", app)

slack_web_client = WebClient(token=SLACK_BOT_TOKEN)

@slack_event_adapter.on("message")
def handle_message(event_data):
    message = event_data["event"]

    # Skip bot's own message
    if message.get('subtype') == 'bot_message':
        return

    # Get the ID of the channel where the message was sent
    channel_id = message["channel"]

    # Get the text of the message that was sent
    message_text = message.get('text')

    # Process the message text and get the response
    response = handle_single_message(message_text)

    # Send the response back to the channel
    try:
        slack_web_client.chat_postMessage(channel=channel_id, text=response)
    except SlackApiError as e:
        print(f"Error sending message: {e}")

if __name__ == "__main__":
    app.run(port=3000)
