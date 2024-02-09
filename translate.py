import requests
from googletrans import Translator
import os

# Specify the Hugging Face API URL
api_url = "https://datasets-server.huggingface.co/first-rows?dataset=SciPhi%2FAgentSearch-V1&config=default&split=train"

# Specify the folder path to save the Parquet files
folder_path = os.path.join(os.getcwd(), 'dataset')
print(folder_path)

file_counter = 1

base_folder_name = os.path.basename(os.path.dirname(folder_path))

# Batch translation parameters
batch_size = 10
sentences_batch = []

def start_translation():
    global file_counter
    translator = Translator()  # Create a Translator instance

    # Fetch data from Hugging Face API
    response = requests.get(api_url)

    # Check if the response is a JSON (dictionary) before proceeding
    if response.status_code == 200 and isinstance(response.json(), dict):
        data_json = response.json()
        data_json = data_json["rows"]
        print(data_json[0])
    else:
        print(f"Failed to fetch valid data from Hugging Face API. Status code: {response.status_code}")
        return

    # Loop through each row in the dataset
    for row_index, row_data in data_json.items():
        if "text-chunks" in row_data:
            text_chunks = row_data["text-chunks"]

            sentences_batch.append(text_chunks)

            if len(sentences_batch) == batch_size:
                translate_and_write_batch(translator)

    # Check if there are remaining sentences in the batch
    if sentences_batch:
        translate_and_write_batch(translator)

    print("Translation ended.")

def translate_and_write_batch(translator):
    try:
        # Translate the batch of sentences
        hindi_sentences_batch = translate_to_hindi_batch(sentences_batch, translator)

        # Save the translated sentences to a file
        output_filename = f'{base_folder_name}_{file_counter}.txt'
        with open(output_filename, 'w', encoding='utf-8') as output_file:
            output_file.write('\n'.join(hindi_sentences_batch) + '\n')

        # Clear the batch for the next set of sentences
        sentences_batch.clear()

        file_counter += 1
    except Exception as e:
        print(f"An error occurred during batch translation: {e}")

def translate_to_hindi_batch(sentences, translator):
    try:
        # Translate the batch of sentences
        translated_sentences = translator.translate(sentences, src='en', dest='hi')
        return [sentence.text for sentence in translated_sentences]
    except Exception as e:
        print(f"An error occurred during batch translation: {e}")
        return [""] * len(sentences)

if __name__ == "__main__":
    start_translation()
