#!/usr/bin/env python3
# utils.py - Helper functions for the chatbot

import random

def get_response(model, message):
    tag = model.predict_tag(message)
    for intent in model.intents["intents"]:
        if intent["tag"] == tag:
            return intent["responses"][0]  # Return first response for simplicity
    return "I'm sorry, I didn't understand that."

