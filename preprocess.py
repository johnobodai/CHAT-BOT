#!/usr/bin/env python3
# preprocess.py - Preprocess data for the chatbot model

import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder

def preprocess_data(intents):
    patterns = []
    tags = []
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            patterns.append(pattern)
            tags.append(intent['tag'])
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(patterns).toarray()
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(tags)

    return X, y, vectorizer, encoder

