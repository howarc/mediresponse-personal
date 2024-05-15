from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel
from torch import cuda
import language_tool_python
from autocorrect import Speller
import re

model = GPT2DoubleHeadsModel.from_pretrained('./GPT2_MediResponse/SaveFile')
tokenizer = GPT2Tokenizer.from_pretrained('./GPT2_MediResponse/SaveFile')

device = 'cuda' if cuda.is_available() else 'cpu'
model = model.to(device)

assert model.transformer.wte.weight.size(0) == len(tokenizer)

def generate_text(prompt, max_length=80):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    chat_history_ids = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty= 1.1,
        do_sample = True,
        temperature=0.9,  
        top_p=0.9,        
        top_k=50,          
        no_repeat_ngram_size=2
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

prompt1 = "[BOS] [PERSONA] You are a relative of a hospitalized patient. The patient is in critical condition. You are feeling anger. "
prompt2 = "[DOC] I understand your concern. It's a tough situation. Unfortunately, their condition has worsened. [PATIENT] "
response = generate_text(prompt1 + prompt2)

# cleaning output
def clean_output(input_string):
    input_string = re.sub(r'\s*([?,.!"])\s*', r'\1 ', input_string) # ensure space after punc

    input_string = input_string.replace('"', "") # remove "

    # input_string = re.sub(r'\s*([?,.!])\s*', r'\1 ', input_string)

    input_string = input_string.strip() # strip leading and trailing whitespace before final clean-up

    # Ensure a single space between words and after punctuation
    input_string = re.sub(r'\s{2,}', ' ', input_string)
    
    # trim text to last punctuation
    m = re.search(r'([.!?])[^.!?]*$', input_string)
    if m:
        input_string = input_string[:m.start()+1]

    return input_string
    
response = clean_output(response)

# grammar

spell = Speller()
corrected_text = spell(response)

print("With Grammar:", corrected_text)
print("Without:", response)