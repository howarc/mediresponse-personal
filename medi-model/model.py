from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch import cuda
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# init
device = 'cuda' if cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  

model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.resize_token_embeddings(len(tokenizer))  

# preprocess
def preprocess_data(file_path, output_file):
    df = pd.read_csv('./dataset/mediresponse.csv', skipinitialspace=True)
    df['dialogue'] = df['Input'].astype(str) + " <|endoftext|> " + df['Target'].astype(str)
    df['dialogue'].to_csv(output_file, header=False, index=False, sep="\n")

    train, test = train_test_split(df['dialogue'], test_size=0.3, random_state=42)
    
    # Save train and test sets to files
    train.to_csv('./dataset/train_dataset.txt', header=False, index=False, sep="\n")
    test.to_csv('./dataset/test_dataset.txt', header=False, index=False, sep="\n")



preprocess_data('./dataset/mediresponse.csv', './dataset/preprocessed_conversation.txt')

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
    output_dir='./GPT2_MediResponse',
    overwrite_output_dir=True,
    num_train_epochs=5,
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
model.save_pretrained('./GPT2_MediResponse/SaveFile')
tokenizer.save_pretrained('./GPT2_MediResponse/SaveFile')

def generate_text(prompt, max_length=100):
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

prompt1 = "A patient has been hospitalized because of a severe burn. "
prompt2 = "The doctor tells you they are currently in critical condition. "
prompt3 = "As the talkative, fretful son of this patient, you ask: "
response = generate_text(prompt1 + prompt2 + prompt3)
print("Generated Response:", response)


## Structure of a situation 
# A patient has been hospitalized because of a [sports injury / car accident / heart attack / severe allergic reaction / severe burns / stroke ]
# "You are [his/her/their] [friend/ parent/ grandparent/ boyfriend / girlfriend/ spouse / fiance / roommate / child / grandchild]."
# As the doctor, you know their condition is [stable / critical / improving / detiorarting]/[passed away].
##