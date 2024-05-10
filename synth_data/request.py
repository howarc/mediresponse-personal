import google.generativeai as genai
import pandas as pd 
import re
import time
import os
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def remove_non_standard_characters(text):
  pattern = r'[^a-zA-Z0-9",.\'\s]'
  cleaned_text = re.sub(pattern, '', text)
  return cleaned_text

"""
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
"""

model = genai.GenerativeModel('gemini-1.5-pro-latest')
data = pd.read_csv('data.csv', skipinitialspace=True)
data_string = data.to_string(line_width=None, index=False)

prompt = data_string[0:300]
prompt += 'Continue from this dataset. Prompt: Create a CSV dataset capturing emotional conversations between a doctor and a relative/friend of a hospitalized patient. No need to include the header in your response. Each conversation should consist of 5 paired exchanges between the doctor and relative/friend, focusing on the emotional aspect with minimal medical terminology. The dataset should include 20 unique scenarios with these details: Headers: Input (Doctors Statement), Target (Relatives Response), Progression (Stress Level: N for Neutral, D for Decreasing, I for Increasing) These stress levels should be chars and not strings. Content Requirements: Verify the CSV format, with paired exchanges on the same row and no irregular line breaks. Please ensure that each exchange pair (doctors statement and relatives response) remains on the same line within the CSV, separated by a comma. Emotional intensity should be captured in the relatives/friends responses, progressing through various stress levels over the 5 exchanges. Each scenarios 5 exchanges should have a consistent overarching theme, but vary significantly between the 20 different scenarios. This means a total of 100 lines of data. No repetition of exact phrases or replicated scenarios is allowed across the entire dataset. All dialogue lines should be enclosed in quotes to maintain clarity and formality. Output Format: The dataset should be formatted properly for training an NLP model, with no informal symbols or unexpected characters. Generate around 100 total lines of paired doctor/relative exchanges, constituting 20 distinct multi-exchange scenarios. Ensure diverse emotional tones and logical progression of stress levels within each scenarios exchanges. Aim for realistic, contextually appropriate conversations without overly specific personal details (no names). Double check that all 20 scenarios are completely unique with no unintended repetition. Note, you often start to separate these responses towards the end of the conversation, make sure this does not happen. Generate the full 20 scenario CSV dataset adhering to these instructions. Note, I do not want a sample, an example, or explanation, I want the entire CSV dataset.'

file = open('test_data.csv', 'a')
iterations = 100 # change iterations based on how many lines you want

for i in range(iterations):
  response = model.generate_content(prompt).text
  cleaned_response = remove_non_standard_characters(response)
  file.write(cleaned_response)
  time.sleep(5)

