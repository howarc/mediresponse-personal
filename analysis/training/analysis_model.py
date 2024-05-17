import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader
from torch import cuda
import sys
sys.path.append('../../utils')
import utils


device = 'cuda' if cuda.is_available() else 'cpu'
print(device)

df_train = pd.read_csv('../../data/train.txt', sep=';')
df_test = pd.read_csv('../../data/test.txt', sep=';')
df_eval = pd.read_csv('../../data/val.txt', sep=';')

df_train.columns = ['text', 'emotion']  
df_test.columns = ['text', 'emotion']
df_eval.columns = ['text', 'emotion']

df_combined_test = pd.concat([df_test, df_eval]).reset_index(drop=True)
print(df_combined_test.columns)
# be careful and make sure to reset indices for both dataframes
df_train['emotion'] = pd.get_dummies(df_train['emotion']).values.tolist()
df_combined_test['emotion'] = pd.get_dummies(df_combined_test['emotion']).values.tolist()

df_train['emotion'] = df_train['emotion'].apply(lambda x: [int(i) for i in x]).reset_index(drop=True)
df_combined_test['emotion'] = df_combined_test['emotion'].apply(lambda x: [int(i) for i in x]).reset_index(drop=True)


MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
CustomDataset = utils.CustomDataset
training_set = CustomDataset(df_train, tokenizer, MAX_LEN, 'text', 'emotion') 
testing_set = CustomDataset(df_combined_test, tokenizer, MAX_LEN, 'text', 'emotion')

train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }


training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.3) 
        self.l3 = torch.nn.Linear(768, 6) 
    
    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output
    
model = BERTClass()
model.to(device)

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def train(epoch):
    model.train()
    for batch in training_loader:
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
        targets = batch['labels'].to(device, dtype = torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)

        print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(EPOCHS):
    train(epoch)
torch.save(model.state_dict(), '../models/bert_model.pth')

#model.load_state_dict(torch.load('bert_model.pth'))

def validation():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for batch in testing_loader:
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
            targets = batch['labels'].to(device, dtype = torch.float)
            outputs = model(ids, mask, token_type_ids)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets

outputs, targets = validation()
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f'Accuracy: {accuracy}')
print(f'F1 Score (Micro): {f1_score_micro}')
print(f'F1 Score (Macro): {f1_score_macro}')
