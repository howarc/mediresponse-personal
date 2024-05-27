from response import relative_response, get_model_and_tokenizer
import random

# load pre-trained model and tokenizer
base_model, tokenizer = get_model_and_tokenizer()

# prompt line 1 of 2
emotion = ["anger", "fear", "sadness", "surprise"]
chosen_emotion = random.choice(emotion) 
prompt1 = "[BOS] [PERSONA] You are a relative of a hospitalized patient. The patient is in critical condition. You are feeling "
prompt1 = prompt1 + chosen_emotion + ". "

# prompt line 2 of 2
setup = ["It is looking bad.", "We are doing our best."]
chosen_setup = random.choice(setup)
prompt2 = "[DOC] Your relative is in critical condition. "
prompt2 = prompt2 + chosen_setup + "[PATIENT] "

response = relative_response(prompt1 + prompt2)

print("Setting: An intern is having trouble talking to a relative of a hospitalized patient. Hearing the commotion, you step in for the intern.")
print("Relative: " + response)

for i in range(4):
    doc_response = input("Doctor (You): ")
    prompt2 = "[DOC] " + doc_response + " [PATIENT] "
    response = relative_response(prompt1 + prompt2)
    print("Relative: " + response)
