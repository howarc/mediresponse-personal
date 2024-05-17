from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertModel
from torch import cuda
from autocorrect import Speller
import re
<<<<<<< HEAD
import wordninja
import random
=======
import sys
sys.path.append('../utils') 
import utils

#utils.classify('emotion', 'I am feeling very sad today.')
>>>>>>> ec04b3532645215430da2a3eb79c4f45c6361d9c

import sys
sys.path.append('/analysis/models/bert_model_role.pth')
import classify

# LOAD GPT2 FROM SAVE FILE
model = GPT2LMHeadModel.from_pretrained('./GPT2_MediResponse/SaveFile')
tokenizer = GPT2Tokenizer.from_pretrained('./GPT2_MediResponse/SaveFile')
device = 'cuda' if cuda.is_available() else 'cpu'
model = model.to(device)

# GENERATE TEXT
def generate_text(prompt, max_length=60):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    chat_history_ids = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty= 1.005,
        do_sample = True,
        temperature=0.2,  
        top_p=0.88,        
        top_k=56,          
        no_repeat_ngram_size=2
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# emotion = ["anger", "fear", "sadness", "surprise"]
# chosen_emotion = random.choice(emotion) 
# prompt1 = "[BOS] [PERSONA] You are a relative of a hospitalized patient. The patient is in critical condition. You are feeling "
# prompt1 = prompt1 + chosen_emotion + ". "


# setup = ["It is looking bad.", "We are doing our best."]
# chosen_setup = random.choice(setup)
# prompt2 = "[DOC] Your relative is in critical condition. "
# prompt2 = prompt2 + chosen_setup + "[PATIENT] "

# chat_history = ""

prompt1 = "[BOS] [PERSONA] You are a relative of a hospitalized patient. The patient is in critical condition. You are feeling anger. "
prompt2 = "[DOC] Your relative is in critical condition. It is looking bad. [PATIENT] "

# cleaning output
def clean_output(input_string):
    input_string = input_string.replace('"', "") # remove "
    input_string = input_string.replace('. . .', '...') # reformat ...
    
    input_string = re.sub(r'\s*([?,.!"])\s*', r'\1 ', input_string) # ensure space after punc

    input_string = input_string.strip() # leading and trailing whitespace

    input_string = re.sub(r'\s{2,}', ' ', input_string) # single space between words and after punctuation
    
    # punc before capitalized letter
    words = input_string.split()
    corrected_words = []
    for i, word in enumerate(words):
        if (word[0].isupper() and i != 0 and not words[i-1][-1] in '.!?'):
            # check to avoid adding a period if the previous word ends with certain punctuation
            if not words[i-1][-1] in ',:;':
                corrected_words[-1] += '.'
        corrected_words.append(word)
    input_string = ' '.join(corrected_words)
    
    # trim text to last punctuation
    m = re.search(r'([.!?])[^.!?]*$', input_string)
    if m:
        input_string = input_string[:m.start()+1]

    return input_string

def relative_response(input_string):
    response = generate_text(input_string) # initial output from model

    response = wordninja.split(response) # sometimes multiple words are joined together, we split
    response = ' '.join(response) # rejoin to make proper sentence

    response = clean_output(response) # further clean output, see function above

    # spelling of singluar words/autocorrect
    spell = Speller()
    response = spell(response)

    return response

response = relative_response(prompt1 + prompt2)
# chat_history += response + " [DOC] "
# print("Relative:" + response)

# print("You encounter a relative of a hospitalized patient who has been recently informed about their critical condition. Upon hearing this news, they feel anger.")

    