import json
from transformers import Trainer, TrainingArguments
from utils import preprocess_data, get_dataset

def main():
    with open("data/intents.json") as f:
        intents = json.load(f)
    
    # Process and load dataset
    dataset = get_dataset(intents)
    
    # Training configuration
    training_args = TrainingArguments(
        output_dir="./models",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs'
    )
    
    # Initialize Trainer and train the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )
    trainer.train()

if __name__ == "__main__":
    main()

