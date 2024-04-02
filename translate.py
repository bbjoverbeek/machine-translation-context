import pandas as pd
import numpy as np
import pprint
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import load_from_disk, concatenate_datasets
import os
import argparse


def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str,
                        help="Language to translate to.",)
    parser.add_argument("--save_path", type=str,
                        help="Path to save the annotated dataset.",)

    args = parser.parse_args()
    return args


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
    translated_sentence = [en_to_lang(model=model, tokenizer=tokenizer, en_sentence=s, lang=lang) for s in row['context']['en']] 
    row['context'][lang] = translated_sentence
    return row

if __name__ == '__main__':
    args = create_arg_parser()

    dataset = load_dataset()
    
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    
    dataset = dataset.map(add_translations, 
                                      fn_kwargs={'model': model, 'tokenizer': tokenizer, 'lang': args.lang})
    
    dataset.save_to_disk(f'{args.save_path}/')
    dataset.to_csv(f'{args.save_path}.csv')
