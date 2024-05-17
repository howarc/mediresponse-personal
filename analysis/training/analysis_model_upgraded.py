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

emotion_files = {
    'anger': '../../medi-model/dataset/emotions/anger.csv',
    'fear': '../../medi-model/dataset/emotions/fear.csv',
    'joy': '../../medi-model/dataset/emotions/joy.csv',
    'sadness': '../../medi-model/dataset/emotions/sadness.csv',
    'surprise': '../../medi-model/dataset/emotions/surprise.csv'
}

emotions = ['anger', 'fear', 'joy', 'sadness', 'surprise']
dfs = []

for emotion in emotions:
    path = emotion_files[emotion]
    df = pd.read_csv(path)
    df['emotion'] = emotion
    df = df.drop(['Input', 'Progression'], axis=1)
    dfs.append(df)

combined_df = pd.concat(dfs).reset_index(drop=True)

combined_df['emotion'] = pd.get_dummies(combined_df['emotion']).values.tolist()
combined_df['emotion'] = combined_df['emotion'].apply(lambda x: [int(i) for i in x]).reset_index(drop=True)

combined_df = combined_df.sample(frac=1).reset_index(drop=True)
print(combined_df.head())

MAX_LEN = 200
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    

CustomDataset = utils.CustomDataset
train_size = 0.8
train_dataset = combined_df.sample(frac=train_size, random_state=27)
test_dataset = combined_df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN, 'Target', 'emotion')
testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, 'Target', 'emotion')


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
        self.l3 = torch.nn.Linear(768, 5) 
    
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
torch.save(model.state_dict(), '../models/bert_model_upgrade.pth')

#model.load_state_dict(torch.load('bert_model_upgrade.pth'))

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
