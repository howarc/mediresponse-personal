import numpy as np
import pandas as pd
import transformers
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'
print(device)
class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.max_len = max_len

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        }
    
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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model.load_state_dict(torch.load('bert_model.pth'))
model.eval()
max_len = 512
BATCH_SIZE = 4


def check_emotion(responses): # note: responses is a list of strings
    predictions = []
    for response in responses:
        df = pd.DataFrame([response], columns=['text']) # convert the string to a dataframe
        df['text'] = df['text'].apply(lambda x: x.lower()) # convert each sentence to lowercase

        sentence_data = CustomDataset(df, tokenizer, max_len)
        sentence_loader = DataLoader(sentence_data, batch_size=BATCH_SIZE, num_workers=0) # create dataloaders for the sentences

        for batch in sentence_loader: # iterate through the dataloader and predict
            ids = batch['ids'].to(device, dtype = torch.long)
            mask = batch['mask'].to(device, dtype = torch.long)
            token_type_ids = batch['token_type_ids'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)
            predictions.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())

    predictions = (np.array(predictions) >= 0.5).astype(int)
    return predictions

#testArry = ["I am so happy", "I am so sad", "I am so surprised", "I am so angry", "I am so scared", "I am so loving"]
#print(check_emotion(testArry))

# Note: 'anger' (0), 'fear' (1), 'joy' (2), 'love' (3), 'sadness' (4), 'surprise' (5) is the order of the labels
