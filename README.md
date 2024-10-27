# Chatbot Application - Odai

## Overview

Odai is a conversational chatbot designed to assist users in the domain of mental health. Utilizing advanced natural language processing techniques, Odai provides support, information, and conversation to help users navigate their feelings and experiences. The chatbot is built using TensorFlow and incorporates state-of-the-art machine learning models to ensure meaningful and empathetic interactions.

## Features

- **Intelligent Conversation:** Engages users in friendly conversation, responding to greetings, inquiries, and emotional expressions.
- **Emotion Recognition:** Understands user inputs and responds appropriately based on detected sentiment.
- **User-Friendly Interface:** Simple command-line interface that allows for easy interaction and clear output display.
- **Personalized Responses:** Delivers customized responses based on user patterns and predefined intents.
- **Extensive Preprocessing:** Implements thorough preprocessing steps, including tokenization and normalization, to improve the accuracy of user interactions.
- **Hyperparameter Tuning:** Optimizes model performance through tuning of various hyperparameters, resulting in improved validation metrics.

## Installation

### Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x installed on your machine.
- Access to Google Colab or a local development environment with TensorFlow and other required libraries.

### Libraries

You can install the necessary libraries using pip. If you are using Google Colab, the libraries can be installed directly within the notebook:

```bash
!pip install tensorflow pandas sklearn
```

## Usage

### Getting Started

1. **Clone the Repository:**
   Clone this repository to your local machine or open it in Google Colab.

2. **Mount Google Drive (for Colab Users):**
   If you are using Google Colab, ensure you mount your Google Drive to access the chatbot files:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Data Preparation:**
   Place your `intents.csv` file containing user intents and responses in the specified directory:
   `/content/drive/MyDrive/CHAT-BOT/intents.csv`.

4. **Run the Training Script:**
   Execute the training script to train the chatbot model or just run the notebook which contains every code:


5. **Interact with the Chatbot:**
   You can create a separate script or function to allow users to interact with the trained model. This script will load the model and process user input, generating responses based on user queries.

### Example Interaction

Upon running the interaction script, users can input phrases like:
- "Hi"
- "How are you?"
- "What should I call you?"

Odai will respond with thoughtful replies based on the user’s input.

## Model Evaluation

The model is evaluated using various metrics, including:
- **Accuracy:** Measures the proportion of correct predictions.
- **Precision:** Assesses the accuracy of positive predictions.
- **Recall:** Evaluates the model's ability to find all relevant instances.

## Future Enhancements

- **GUI Development:** Implement a graphical user interface to improve user interaction.
- **Multi-language Support:** Expand capabilities to support multiple languages.
- **Continuous Learning:** Allow the chatbot to learn from user interactions for improved responses.

## Contribution

Contributions to the project are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.
