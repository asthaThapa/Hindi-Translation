import torch
from transformers import (
    LlamaForCausalLM, LlamaConfig, LlamaTokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset
import sentencepiece as spm
import os
import logging
import json
import sys
import argparse

USER_DEFINED_SYMBOLS = ["<pad>", "<s>", "</s>", "<mask>", "."]
symbols = USER_DEFINED_SYMBOLS

vowels =[ "अ", "आ", "इ",  "ई",  "उ", "ऊ", "ऋ","ॠ", "ए", "ऐ","ओ", "औ", "अं", "अः"]

consonants = [
  "क", "ख", "ग", "घ", "ङ",
  "च", "छ", "ज", "झ", "ञ",
  "ट", "ठ", "ड", "ढ", "ण",
  "त", "थ", "द", "ध", "न",
  "प", "फ", "ब", "भ", "म",
  "य", "र", "ल", "व",
  "श", "ष", "स", "ह",
  "क्ष", "त्र", "ज्ञ"
]

dependent_vowels = [
  "ा", # aa
  "ि", # i
  "ी", # ii
  "ु", # u
  "ू", # uu
  "ृ", # ri
  "ॄ", # rri
  "े", # e
  "ै", # ai
  "ो", # o
  "ौ", # au
  "ं", # an
  "ः"  # ah
]

symbols.extend(vowels + consonants + dependent_vowels)

print(symbols)

def train_tokenizer(language, input_path, model_prefix, vocab_size):
    spm.SentencePieceTrainer.train(
        input=input_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        user_defined_symbols=language,
        model_type="BPE"
    )

def move_tokenizer_to_folder(source, destination_folder):
    os.rename(source, os.path.join(destination_folder, "tokenizer.model"))

def create_config_file(folder_path, content):
    with open(os.path.join(folder_path, "config.json"), "w") as config_file:
        json.dump(content, config_file, indent=4)

config_content = {
    "_name_or_path": "./names_1m",
    "architectures": [
        "LlamaForCausalLM"
    ],
    "bos_token_id": 2,
    "eos_token_id": 3,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 180,
    "max_position_embeddings": 32,
    "model_type": "llama",
    "num_attention_heads": 16,
    "num_hidden_layers": 8,
    "num_key_value_heads": 16,
    "pad_token_id": 1,
    "pretraining_tp": 1,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "tie_word_embeddings": False,
    "torch_dtype": "float32",
    "transformers_version": "4.28.1",
    "use_cache": False,
    "vocab_size": 73
}


out_folder_path = "names"
os.makedirs(out_folder_path, exist_ok=True)
create_config_file(out_folder_path, config_content)
train_tokenizer(symbols, 'data.txt', 'tokenizer', 73)
move_tokenizer_to_folder("tokenizer.model", out_folder_path)

tokenizer = LlamaTokenizer.from_pretrained(out_folder_path)
tokenizer.pad_token = tokenizer.eos_token

sample_sentence = "आर्या गुप्ता"
tokens = tokenizer(
                sample_sentence, truncation=True,
                padding='max_length', max_length=16)
print(f"Original Sentence: {sample_sentence}\nTokenized Sentence: {tokens}")

tokenizer.decode([1, 69, 7, 46, 72, 45, 56, 69, 22, 59, 40, 72, 35, 56, 2, ])

def create_config_model(path):
    config = LlamaConfig.from_pretrained(path)

    model = LlamaForCausalLM(config)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_size = sum(t.numel() for t in model.parameters())
    print(f"GPT Model size: {model_size/1000**2:.1f}M parameters")
    return model

def create_tokenized_dataset_splits(path, tokenizer, block_size):
    dataset = load_dataset('text', data_files=path)
    shuffled_dataset = dataset['train'].shuffle(seed=5)
    split_datasets = shuffled_dataset.train_test_split(test_size=0.2)

    def tokenize_dataset(dataset):
        return dataset.map(
            lambda examples: tokenizer(
                examples['text'], truncation=True,
                padding='max_length', max_length=block_size
            ),
            batched=True
        )

    def unique_name_set(dataset):
      names_set = set()

      for example in dataset:
          name = example['text'].split(".")[0]
          names_set.add(name)

      return names_set

    return tokenize_dataset(split_datasets['train']), tokenize_dataset(split_datasets['test']), unique_name_set(split_datasets['train'])


def train_model(model, tokenizer, train_dataset, test_dataset, out_folder_path):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=out_folder_path,
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=8,
        save_steps=10000,
        logging_steps=10,
        eval_steps=1000,
        logging_dir=f'{out_folder_path}/logs',
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2, early_stopping_threshold=0.001)]
    )

    trainer.train()
    model.save_pretrained(out_folder_path)


model = create_config_model(out_folder_path)
train_dataset, test_dataset, unique_names = create_tokenized_dataset_splits('data.txt', tokenizer, block_size=32)
train_model(model, tokenizer, train_dataset, test_dataset, out_folder_path)

def generate_names(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids).to(model.device)
    generated_names = set()

    with torch.no_grad():
        while len(generated_names) < 20:
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=32,
                early_stopping=True,
                temperature=0.6,
                top_p=0.8,
                top_k=50,
                do_sample=True,
                output_scores=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.4,
                eos_token_id=tokenizer.eos_token_id
            )
            output_str = tokenizer.decode(output[0], skip_special_tokens=True).split(".")[0]
            if output_str not in generated_names and output_str not in unique_names:
                print(output_str)
                generated_names.add(output_str)

male_names_prompt = "लड़का,"
female_names_prompt = "लड़की,"

model.eval()
generate_names(model, tokenizer, male_names_prompt)
generate_names(model, tokenizer, female_names_prompt)