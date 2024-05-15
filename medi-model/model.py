from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch import cuda
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


device = 'cuda' if cuda.is_available() else 'cpu'

# tokens
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
new_tokens = ['[BOS]', '[PERSONA]', '[DOC]', '[PATIENT]']  
all_special_tokens = {**tokenizer.special_tokens_map, **{'additional_special_tokens': new_tokens}}
tokenizer.add_special_tokens(all_special_tokens)
tokenizer.pad_token = tokenizer.eos_token  

# model
model = GPT2DoubleHeadsModel.from_pretrained('gpt2')
model.to(device)
model.resize_token_embeddings(len(tokenizer))  

def preprocess_data():
    folder_path = './dataset/emotions/'

    # get a list of all CSV files in the folder
    file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

    combined_df = pd.DataFrame()

    for file_path in file_paths:
        # extract emotion from the filename
        emotion = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path, skipinitialspace=True)

        # Create a dialogue column formatted with the emotion token
        df['dialogue'] = f"[BOS] [PERSONA] You are a relative of a hospitalized patient. The patient is in critical condition. You are feeling {emotion}. [DOC] " + df['Input'].astype(str) + " [PATIENT] " + df['Target'].astype(str) + " <|endoftext|>"

        combined_df = pd.concat([combined_df, df['dialogue']], ignore_index=True)

    train, test = train_test_split(combined_df, test_size=0.3, random_state=42)
    train.to_csv('./dataset/train_dataset.txt', header=False, index=False, sep="\n")
    test.to_csv('./dataset/test_dataset.txt', header=False, index=False, sep="\n")


preprocess_data()

train_path = './dataset/train_dataset.txt'
test_path = './dataset/test_dataset.txt'

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
    # parameters
    num_train_epochs=16,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    eval_steps=16000,
    save_steps=1600, 
    warmup_steps=1600,
    weight_decay=1,
    learning_rate=1e-5,  
    
    # logging
    output_dir='./GPT2_MediResponse',
    overwrite_output_dir=True,
    
    # efficency purposes
    fp16=True, 

    # loading final model
    # load_best_model_at_end=True, 
    # save_total_limit=3,  
    # evaluation_strategy="steps", 
    # save_strategy="steps", 
    # metric_for_best_model='eval_loss',
    # greater_is_better=False,
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
model.save_pretrained('./GPT2_MediResponse/SaveFile')
tokenizer.save_pretrained('./GPT2_MediResponse/SaveFile')

def generate_text(prompt, max_length=80):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    chat_history_ids = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty= 1.1,
        # do_sample = True,
        # temperature=0.7,  
        # top_p=0.9,        
        # top_k=50,          
        no_repeat_ngram_size=2
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

prompt1 = "[BOS] [PERSONA] You are a relative of a hospitalized patient. The patient is in critical condition. You are feeling anger. "
prompt2 = "[DOC] The next 24 hours are critical. We're closely monitoring their progress. We're doing our best. [PATIENT] "
response = generate_text(prompt1 + prompt2)
print("Generated Response:", response)

