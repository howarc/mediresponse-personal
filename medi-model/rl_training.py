# transformer reinforcement learning

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from trl import PPOTrainer, PPOConfig, create_reference_model, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from convo import relative_response, get_model_and_tokenizer
import random
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*resume_download.*")

def get_human_feedback():
    rating = input("Evaluate the model's response from -2 to 2: ")
    return int(rating)

# load pre-trained model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_model, tokenizer = get_model_and_tokenizer()
model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
model_ref = create_reference_model(model)

# dataset
df = pd.read_csv('./dataset/emotions/anger.csv')
queries = df['Input'].tolist()
responses = df['Target'].tolist()

class DialogueDataset(Dataset):
    def __init__(self, tokenizer, queries, responses):
        self.tokenizer = tokenizer
        self.queries = queries
        self.responses = responses
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]
        encoded_query = self.tokenizer(query, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        encoded_response = self.tokenizer(response, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
        return encoded_query['input_ids'].squeeze(0), encoded_response['input_ids'].squeeze(0)
    
dataset = DialogueDataset(tokenizer, queries, responses)


# init trainer
ppo_config = PPOConfig(batch_size=1, mini_batch_size=1)
ppo_trainer = PPOTrainer(ppo_config, model, model_ref, tokenizer, dataset = dataset)


# training loop
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

data_list = list(data_loader)
random.shuffle(data_list)


for batch in data_list[:5]:
    query_tensor, expected_response_tensor = batch
    query_tensor = query_tensor.to(device)
    query_text = tokenizer.decode(query_tensor[0], skip_special_tokens=True)
    print(f"Doctor: {query_text}")

    expected_response_tensor = expected_response_tensor.to(device)

    # generate model response
    response_tensor = respond_to_batch(model, query_tensor)
    # generated_response = tokenizer.decode(response_tensor[0], skip_special_tokens=True)
    generated_response = relative_response("[DOC] " + query_text + " [PATIENT] ")
    print(f"Generated response: {generated_response}")


    # Get human feedback
    reward = get_human_feedback()
    reward_tensor = torch.tensor([reward], dtype=torch.float32).to(device)

    # Train model for one step with PPO
    train_stats = ppo_trainer.step([query_tensor[0]], [response_tensor[0]], [reward_tensor])
    # print(f"Training stats: {train_stats}")

model.save_pretrained('./training/PPO_trained')
tokenizer.save_pretrained('./training/PPO_trained')