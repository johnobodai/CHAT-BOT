#!/usr/bin/env python3
import json
from model import ChatBotModel
from utils import get_response

def main():
    # Load intents and initialize the chatbot model
    with open('data/intents.json', 'r') as file:
        intents = json.load(file)
    
    chatbot = ChatBotModel(intents)
    print("Welcome to the Mental Health Chatbot!")
    print("I'm Odai. I am a conversational agent designed to mimic a therapist.")
    print("So, how are you feeling today? Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        
        if user_input.lower() == 'quit':
            print("Odai: Thank you for chatting. Take care!")
            break
        
        response = get_response(chatbot, user_input)
        print(f"Odai: {response}")

if __name__ == "__main__":
    main()
