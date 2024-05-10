import re
import pandas as pd

def process_csv(input_file, output_file):
    
    with open(input_file, 'r') as f:
        data = f.read()

    data = re.sub(r'\s*,\s*', ',', data) # removes all spaces around commas
    
    with open('temp.csv', 'w') as f:
        f.write(data)
    
    df = pd.read_csv('temp.csv', header=None, on_bad_lines='skip')
    df = df[df.count(axis=1) >= 3] # removes rows with less than 3 columns
    df[2] = df[2].str.replace("'", "") # removes all single quotes for the progression column
    df.to_csv(output_file, index=False, header=False)
    
process_csv('test_data.csv', 'processed_data.csv')