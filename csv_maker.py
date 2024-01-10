import os
import csv

current_directory = os.getcwd()
folder_name = 'transcripts'
folder_path = os.path.join(current_directory, folder_name)

csv_file_name = 'conversation_data.csv'

headers = ['Doctor', 'Patient']

# Create a CSV file to write the conversations
with open(csv_file_name, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(headers)  # Write headers to the CSV file
    
    # Check if the folder exists
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        # Loop through each file in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                

                try:
                    with open(file_path, 'r', encoding='utf-8') as txt_file:
                        lines = txt_file.readlines()
                        i = 0

                        while i + 2 < len(lines):
                            patient_message = lines[i].strip().split(': ', 1)[-1]
                            doctor_message = lines[i+2].strip().split(': ', 1)[-1]
                            
                            writer.writerow([patient_message, doctor_message])
                            
                            i += 4

                except UnicodeDecodeError:
                    print(f"Skipping file '{filename}' due to encoding issue.")

    else:
        print(f"Folder '{folder_name}' not found in the current directory.")

print(f"Conversations from .txt files in folder '{folder_name}' converted and saved to {csv_file_name}.")

with open(csv_file_name, 'r', newline='', encoding='utf-8') as file:
    lines = list(csv.reader(file))

    # Filter lines that don't contain only commas
    filtered_lines = [row for row in lines if any(field.strip() != '' for field in row)]

# Overwrite the input CSV file with the filtered content
with open(csv_file_name, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerows(filtered_lines)

