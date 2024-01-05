from googletrans import Translator
import os
import pandas as pd

# Specify the folder path containing Parquet files
folder_path = os.getcwd() + '/'
print(folder_path)

file_counter = 1

base_folder_name = os.path.basename(os.path.dirname(folder_path))


def dict_to_text(data):
    text = ''
    for key, value in data.items():
        text += f'{key}: {value} '
    return text

def list_to_text(data):
    return " ".join(data)    

def start_translation():
    global file_counter
    translator = Translator()  # Create a Translator instance

    # Loop through each Parquet file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.parquet'):
            print(filename)
            file_path = os.path.join(folder_path, filename)

            output_filename = f'{base_folder_name}_{file_counter}.txt'
            with open(output_filename, 'w', encoding='utf-8') as output_file:
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

                        hindi_sentences = translate_to_hindi(cell_value, translator)  # Pass the translator instance

                        if hindi_sentences:
                            output_file.write(hindi_sentences + '\n')

    print("Translation ended.")

                            
def translate_to_hindi(sentence, translator):
    if sentence is None:
        print("Warning: The sentence is None. Skipping translation.")
        return ""

    try:
        translated_sentences = translator.translate(sentence, src='en', dest='hi')
        return translated_sentences.text
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return ""           

if __name__ == "__main__":
    start_translation()
