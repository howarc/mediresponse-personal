from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch import cuda
import torch
import numpy as np
import pandas as pd

# Initialize device and tokenizer
device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

# Load and prepare the model
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.resize_token_embeddings(len(tokenizer))  # Resize model embeddings to account for new tokens

# Preprocess the data
def preprocess_data(file_path, output_file):
    df = pd.read_csv(file_path)
    df['dialogue'] = df['Input (Doctor)'].astype(str) + " <|endoftext|> " + df['Target (Relative)'].astype(str)
    df['dialogue'].to_csv(output_file, header=False, index=False, sep="\n")

# Update your actual paths as necessary
preprocess_data('conversation.csv', 'preprocessed_conversation.txt')

# Set paths to the preprocessed file for both training and testing
train_path = 'preprocessed_conversation.txt'
test_path = 'preprocessed_conversation.txt'

def load_dataset(train_path, test_path, tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)
    
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False)

    return train_dataset, test_dataset, data_collator

train_dataset, test_dataset, data_collator = load_dataset(train_path, test_path, tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=400,
    save_steps=800,
    warmup_steps=500,
    prediction_loss_only=True,
)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained('./doctor_patient_model')
tokenizer.save_pretrained('./doctor_patient_model')

def generate_text(prompt, max_length=100):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    # Generate sequences
    chat_history_ids = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        # no_repeat_ngram_size=2, 
        repetition_penalty= 1.1,  
    )

    # Decode and return generated text
    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# Example usage
prompt = "We're still monitoring them closely."
response = generate_text(prompt)
print("Generated Response:", response)