from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch import cuda
import torch
import numpy as np
import pandas as pd

# init
device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.resize_token_embeddings(len(tokenizer))  

# preprocess
def preprocess_data(file_path, output_file):
    df = pd.read_csv('mediresponse.csv', skipinitialspace=True)
    df['dialogue'] = df['Input (Doctor)'].astype(str) + " <|endoftext|> " + df['Target (Relative)'].astype(str)
    df['dialogue'].to_csv(output_file, header=False, index=False, sep="\n")


preprocess_data('mediresponse.csv', 'preprocessed_conversation.txt')

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

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# save
model.save_pretrained('./doctor_patient_model')
tokenizer.save_pretrained('./doctor_patient_model')

def generate_text(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    chat_history_ids = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty= 1.1,  
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

prompt = "We've stabilized their vitals for now"
response = generate_text(prompt)
print("Generated Response:", response)