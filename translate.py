from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from datasets import load_from_disk, concatenate_datasets, Dataset
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


def en_to_lang(model, tokenizer, en_sentence: str, lang: str) -> str:

    """Translate English context sentences to the target language"""

    tokenizer.src_lang = "en"
    encoded_zh = tokenizer(en_sentence, return_tensors="pt")
    generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id(lang))
    result = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    
    print('\n')
    print('Original sentence:', en_sentence)
    print('Translated sentence:', result)

    fixed_result = input('Manually changed translation: ')

    return fixed_result


def load_dataset(lang: str) -> Dataset:

    """Load dataset"""

    dataset_path = f'annotations/annotations_{lang}'
    dataset = load_from_disk(dataset_path)
    return dataset


def add_translations(row: dict, model, tokenizer, lang: str) -> dict:

    """Add translated context sentences to the dataset"""

    translated_sentence = [en_to_lang(model=model, tokenizer=tokenizer, en_sentence=s, lang=lang) for s in row['context']['en']] 
    row['context'][lang] = translated_sentence
    return row

if __name__ == '__main__':
    
    args = create_arg_parser()

    dataset = load_dataset(lang=args.lang)
    
    model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
    tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
    
    dataset = dataset.map(add_translations, 
                          fn_kwargs={'model': model, 'tokenizer': tokenizer, 'lang': args.lang})
    
    dataset.save_to_disk(f'{args.save_path}/')
    dataset.to_csv(f'{args.save_path}.csv')
