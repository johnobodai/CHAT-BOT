#!/usr/bin/env python3
"""
This script trains a simple neural network model for a chatbot 
using intents data for mental health.
"""

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load intents data
with open('data/intents.json') as file:
    intents = json.load(file)

# Prepare training data
training_sentences = []
training_labels = []
classes = []

for intent in intents:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

# Encode labels
label_encoder = LabelEncoder()
training_labels_encoded = label_encoder.fit_transform(training_labels)

# Tokenize and vectorize the sentences
tokenizer = keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(training_sentences)
X = tokenizer.texts_to_sequences(training_sentences)
X = keras.preprocessing.sequence.pad_sequences(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, training_labels_encoded, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64, input_length=X.shape[1]),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(classes), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=5, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Model accuracy: {accuracy * 100:.2f}%')

# Save the model and tokenizer for future use
model.save('chatbot_model.h5')
with open('tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())
with open('label_encoder.json', 'w') as f:
    f.write(json.dumps(label_encoder.classes_.tolist()))

