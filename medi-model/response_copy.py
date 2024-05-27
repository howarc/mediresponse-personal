from transformers import GPT2Tokenizer, GPT2LMHeadModel, BertTokenizer, BertForSequenceClassification
from sentence_transformers import SentenceTransformer, util
import torch
from torch import cuda
from autocorrect import Speller
import re
import wordninja
import sys
sys.path.append('../utils') 
import utils 
import random

utils.classify('role', 'I know she is. She always has been.')

model = GPT2LMHeadModel.from_pretrained('./GPT2_MediResponse/SaveFile')
sentence_model = SentenceTransformer('nli-roberta-base-v2')
tokenizer = GPT2Tokenizer.from_pretrained('./GPT2_MediResponse/SaveFile')
device = 'cuda' if cuda.is_available() else 'cpu'
model = model.to(device)

# generate deterministic text
def generate_text_det(prompt, max_length=128):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    chat_history_ids = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty= 1.005,
        do_sample = True,
        temperature=0.7,  
        top_p=0.64,        
        top_k=32,          
        no_repeat_ngram_size=2
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

# generate diverse text
def generate_text_div(prompt, max_length=128):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    chat_history_ids = model.generate(
        input_ids,
        max_length=max_length + len(input_ids[0]),
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty= 1.005,
        do_sample = True,
        temperature=0.7,  
        top_p=0.88,        
        top_k=52,          
        no_repeat_ngram_size=2
    )

    response = tokenizer.decode(chat_history_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return response

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

# prompt line 1 of 2
emotion = ["anger", "fear", "sadness"]
chosen_emotion = random.choice(emotion) 
context = "[BOS] [PERSONA] You are a relative of a hospitalized patient. The patient is in critical condition. You are feeling "
context = context + chosen_emotion + ". "

# prompt line 2 of 2
setup = ["It is looking bad.", "We are doing our best."]
chosen_setup = random.choice(setup)
after_context = "[DOC] Your relative is in critical condition. "
after_context = after_context + chosen_setup + "[PATIENT] "

# chat_history = ""

# final response
def relative_response(input_string):
    random_gen = random.choice([0, 1])
    # randomize for diverse or deterministic
    if (random_gen == 0 ):
        response = generate_text_det(input_string) 
    else:
        response = generate_text_div(input_string)

    response = wordninja.split(response) # sometimes multiple words are joined together, we split
    response = ' '.join(response) # rejoin to make proper sentence

    response = clean_output(response) # further clean output, see function above

    # spelling of singluar words/autocorrect
    spell = Speller()
    response = spell(response)

    # now we split response into sentence, and classify if they are actually words of a relative
    sentences = re.split(r'(?<=[.!?])\s+', response)
    responded = False
    final_resp = ""

    for sentence in sentences:
        prediction = utils.classify('role', sentence)
        if prediction[0][1] == 1:
            responded = True
            final_resp += sentence + " "

        else:
            if responded == True:
                break

    final_resp = final_resp[:-1]

    return final_resp

def semantic_filter(cos_sim, nli_pred):
    if (cos_sim > 0.3 and nli_pred == 2) or (cos_sim > 3.5 and nli_pred == 1):
        return True
    else:
        return False


response = relative_response(context + after_context)

print("Setting: You encounter a relative of a hospitalized patient who has been recently informed about their critical condition. Upon hearing this news, they feel " + chosen_emotion + ".")
print("Doctor: Your relative is in critical condition. " + chosen_setup)
print("Relative: " + response)

nli_robert_model = SentenceTransformer("nli-roberta-base-v2")
nli_robert_model.to(device)
nli_bert_tokenizer= BertTokenizer.from_pretrained('textattack/bert-base-uncased-snli')
nli_bert_model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-snli')
nli_bert_model.to(device)

intro = True
for i in range(3):
    doc_response = input("Doctor (You): ")
    after_context = "[DOC] " + doc_response + " [PATIENT] "
    possible_responses = {}

    for i in range(5):
        exit = False
        response = relative_response(context + after_context)
        
        # roberta with cosine similarity
        roberta_premise = nli_robert_model.encode(doc_response, convert_to_tensor=True)
        roberta_hypothesis = nli_robert_model.encode(response, convert_to_tensor=True)

        # filter out responses to reduce computaiton
        if response in possible_responses.values() or response == "" or response == " ":
            exit = True
        
        for key in possible_responses.values():
            premise_check = nli_robert_model.encode(key.get("response"), convert_to_tensor=True)
            if util.pytorch_cos_sim(premise_check, roberta_hypothesis) > 0.85:
                exit = True
                break
        if exit:
            continue
        # bert with nli prediction
        bert_input = nli_bert_tokenizer.encode_plus(doc_response, response, return_tensors='pt', padding=True, truncation=True, max_length=128)
        bert_input = {k: v.to(device) for k, v in bert_input.items()}  # move bert_input to GPU
        entailment = nli_bert_model(**bert_input)
        probs_entailment = torch.nn.functional.softmax(entailment.logits, dim=1)
        predicted_class_entailment = torch.argmax(probs_entailment, dim=1).item()
        possible_responses[i] = {"response": response, "cos_sim": util.pytorch_cos_sim(roberta_premise, roberta_hypothesis), "nli_pred": predicted_class_entailment}

        # we only generate until we have a quality response
        
    for (count, response) in possible_responses.items():
        print("cos_sim: " + str(response.get("cos_sim").item()))
        print("nli_pred: " + str(response.get("nli_pred")))
        print("Relative: " + response.get("response"))
        if semantic_filter(response.get("cos_sim").item(), response.get("nli_pred")):
            print("Final Relative: " + response.get("response"))
            generateBool = False
            break
        else:
            # just print the highest cos_sim and nli_pred
            best_response = max(possible_responses.items(), key=lambda item: item[1].get("cos_sim").item())
            print("Best Response: " + best_response[1].get("response"))
    
