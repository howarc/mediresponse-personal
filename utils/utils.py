import torch
from torch.utils.data import Dataset
import transformers
from transformers import BertTokenizer
from torch import cuda
import numpy as np

class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len, text, targets):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe[text]
        self.targets = self.data[targets]
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
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.long)
        }
    
def classify(model_type, text):
    output_num = 0
    if (model_type == 'emotion'):
        output_num = 6
    elif (model_type == 'role'):
        output_num = 2
    elif (model_type == 'emotion_upgraded'):
        output_num = 5
    else:
        return 'Invalid model type'
    class BERTClass(torch.nn.Module):
        def __init__(self):
            super(BERTClass, self).__init__()
            self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
            self.l2 = torch.nn.Dropout(0.3) 
            self.l3 = torch.nn.Linear(768, output_num) 
        
        def forward(self, ids, mask, token_type_ids):
            _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
            output_2 = self.l2(output_1)
            output = self.l3(output_2)
            return output
        
    device = 'cuda' if cuda.is_available() else 'cpu'
    model = BERTClass()
    model.to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if (model_type == 'emotion'):
        model.load_state_dict(torch.load('../analysis/models/bert_model.pth'))
    elif(model_type == 'emotion_upgraded'):
        model.load_state_dict(torch.load('../analysis/models/bert_model_upgraded.pth'))
    elif (model_type == 'role'):
        model.load_state_dict(torch.load('../analysis/models/bert_model_role.pth'))
    model.eval()
    max_len = 512

    def preprocess_text(text, tokenizer, max_len):
        text = " ".join(text.split())
        inputs = tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )

        ids = torch.tensor(inputs['input_ids'], dtype=torch.long).unsqueeze(0).to(device)
        mask = torch.tensor(inputs['attention_mask'], dtype=torch.long).unsqueeze(0).to(device)
        token_type_ids = torch.tensor(inputs['token_type_ids'], dtype=torch.long).unsqueeze(0).to(device)
        return ids, mask, token_type_ids

    def classify_text(text, model, tokenizer, max_len):
        ids, mask, token_type_ids = preprocess_text(text, tokenizer, max_len)

        with torch.no_grad():
            outputs = model(ids, mask, token_type_ids)
            predictions = torch.softmax(outputs, dim=-1).cpu().numpy().tolist()
        predictions = (np.array(predictions) >= 0.5).astype(int)
        return predictions

    # Doctor, Relative
    input_text = text
    predictions = classify_text(input_text, model, tokenizer, max_len)
    print(predictions)

# If you want to classify text as a role, pass in role
# If you want to classify text as an emotion, pass in emotion
# If you want to classify text as an emotion using the upgraded model, pass in emotion_upgraded