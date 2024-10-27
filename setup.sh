#!/bin/bash

# Create directories
mkdir -p data
mkdir -p models

# Create empty files
touch chatbot.py
touch train.py
touch utils.py
touch requirements.txt
touch README.md

# Populate `requirements.txt` with basic dependencies
echo -e "transformers\ntorch\nnumpy\npandas" > requirements.txt

# Add a basic README template
echo "# Mental Health ChatBot

## Setup
1. Install dependencies: \`pip install -r requirements.txt\`
2. Run \`train.py\` to fine-tune the model.
3. Run \`chatbot.py\` to interact with the chatbot.

## Usage
- Run \`python chatbot.py\` and ask your questions." > README.md

# Confirm setup
echo "Directory structure and files created successfully!"
tree

