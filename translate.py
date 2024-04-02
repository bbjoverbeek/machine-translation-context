import pandas as pd
import numpy as np
import pprint
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import load_from_disk, concatenate_datasets
import os
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    # Type of annotation file to load
    parser.add_argument("--use_csv", action="store_true", 
                        help="Whether to load Dataset file or csv file. Default False.",)
    parser.add_argument("--lang", type=str,
                        help="Language to translate to.",)

    args = parser.parse_args()
    return args


def str_to_dict(string_data):

    # Replace 'array' with 'np.array' in the string
    string_data = string_data.replace('array', 'np.array')

    # Evaluate the string to dictionary
    dictionary_data = eval(string_data, {'np': np})

    # Convert array strings to numpy arrays using the defined function
    for key, value in dictionary_data.items():
        if isinstance(value, np.ndarray):
            dictionary_data[key] = np.array(value)

    return dictionary_data

def read_en_context(filepath):

    dataset = pd.read_csv(filepath)
    context_data = dataset['context']
    context_data = context_data.to_list()
    dict_context_data = [str_to_dict(string_item) for string_item in context_data]

    return dict_context_data


def en_to_lang(model, tokenizer, en_sentence, lang):

    tokenizer.src_lang = "en"
    encoded_zh = tokenizer(en_sentence, return_tensors="pt")
    generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id(lang))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    print('\n')
    print('Original sentence:', en_sentence)
    print('Translated sentence:', result)

    fixed_result = input('Manually changed translation: ')

    return fixed_result


def load_dataset():
    dataset_dirs = ['annotations/'+path for path in os.listdir('annotations/') if not path.endswith('.csv')]
    dataset = concatenate_datasets([load_from_disk(dataset_path) for dataset_path in dataset_dirs])
    return dataset


def add_translations(row, model, tokenizer, lang):
    ru = [en_to_lang(model=model, tokenizer=tokenizer, en_sentence=s, lang=lang) for s in row['context']['en']] 
    row['context']['ru'] = ru
    return row

if __name__ == '__main__':
    args = create_arg_parser()

    if args.use_csv:
        #list of dicts
        all_sentences = read_en_context('annotations/annotations_bjurn_b1_10.csv')
    else:
        all_sentences = load_dataset()
    
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    
    all_sentences = all_sentences.map(add_translations, 
                                      fn_kwargs={'model': model, 'tokenizer': tokenizer, 'lang': args.lang})
    
    # pprint.pprint(all_sentences[0])