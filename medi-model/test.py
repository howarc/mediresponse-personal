import re
import sys
sys.path.append('../utils') 
import utils 

response = ("Help me please. We are doing the best we can.")

sentences = re.split(r'[.!?]\s+', response)
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
print(final_resp)
