from datasets import load_dataset
import pandas as pd

hindi_data_set = load_dataset("text", data_files="path/to/bookcorpus_1.txt")  # Replace "path/to/bookcorpus_1.txt" with the actual path

# Print information about the dataset
print(hindi_data_set)

# Function to get the training corpus
def get_training_corpus(dataset):
    return (
        dataset["train"][i : i + 1000]["text"]
        for i in range(0, len(dataset["train"]), 1000)
    )

# Create a list containing sentences from the training corpus
training_corpus = list(get_training_corpus(hindi_data_set))

# Print the first sentence as an example
print(training_corpus[0])
