import os
import pandas as pd
import re


# Specify the folder path containing Parquet files
folder_path = os.getcwd() + '/'
print(folder_path)

# Function to check if a word is Hindi
def is_hindi(word):
    try:
        print(detect(word) == 'hi')
        return detect(word) == 'hi'
    except:
        # Handle the case where language detection fails
        return False

# Function to convert dictionary to text
def dict_to_text(data):
    text = ''
    for key, value in data.items():
        text += f'{key}: {value} '
    return text

# Function to convert list to text
def list_to_text(data):
    return " ".join(data)    

def extract_hindi_words(text):
    if isinstance(text, str):
        hindi_words = re.findall(r'[ऀ-ॿ]+', text)
        return ' '.join(hindi_words)
    else:
        ''

file_counter = 1

base_folder_name = os.path.basename(os.path.dirname(folder_path))


# Loop through each Parquet file in the folder
for filename in os.listdir(folder_path):
        if filename.endswith('.parquet'):
            print(filename)
            file_path = os.path.join(folder_path, filename)

            # Open the output file outside of the loop
            output_filename = f'{base_folder_name}_{file_counter}.txt'
            with open(output_filename, 'a', encoding='utf-8') as output_file:
                df = pd.read_parquet(file_path)

                # Loop through each row in the DataFrame
                for index, row in df.iterrows():
                
                    for column, cell_value in row.items():
                    
                        if isinstance(cell_value, dict):
                            cell_value = dict_to_text(cell_value)

                        elif isinstance(cell_value, list):
                            cell_value = list_to_text(cell_value)    

                        elif isinstance(cell_value, tuple):
                            cell_value = list_to_text(cell_value)        
    
                        if not isinstance(cell_value, str):
                            try:
                                cell_value = str(cell_value)
                            except TypeError as e:
                                 print(f"An error occurred: {e}")
                
                        hindi_words = extract_hindi_words(cell_value)

                        if hindi_words:
                            output_file.write(hindi_words + '\n')
            file_counter += 1
        

                        
