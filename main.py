# [Your original code here]
...

def handle_single_message(message, chat_history_filename='chat_history.json'):
    # Load the chat history
    chat_history = load_chat_history(chat_history_filename)

    # Run the conversation chain with the user's query and the chat history
    response = conversation_chain.run({"question": message, "chat_history": chat_history})

    # Add the new conversation to the chat history
    chat_history.append({"question": message, "response": response})

    # Save the updated chat history
    save_chat_history(chat_history, chat_history_filename)

    # Return the response
    return response

# [The rest of your original code here]
...
