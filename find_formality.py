"""
Program to filter the IWSLT 2017 Dutch-English dataset for formality clear from context.
It is possible to provide a path to save the filtered dataset to.
"""

import re
import sys
from tqdm import tqdm
from datasets import load_dataset, Dataset


def add_context_sentences(dataset: Dataset) -> Dataset:
    """Adds context (previous 3) sentences to all examples in the dataset.
    Note!: this function takes a sorted, unfiltered dataset as input.
    """

    context = []

    for idx, example in tqdm(
        enumerate(dataset), desc='Adding context sentences', total=len(dataset)
    ):
        context.append({'nl': [], 'en': []})
        for i in range(3, 0, -1):
            # check if the current previous sentence has the same document id
            if example['doc_id'] != dataset[idx - i]['doc_id']:
                continue

            context[idx]['nl'].append(dataset[idx - i]['translation']['nl'])
            context[idx]['en'].append(dataset[idx - i]['translation']['en'])

    return dataset.add_column('context', context)


def find_formality_dutch(
    formality_regex: str, example: dict[str, str | int | dict[str, str]]
) -> bool:
    """Use regular expressions to filter the dataset for formality in Dutch."""

    if re.search(formality_regex, example['translation']['nl'], re.IGNORECASE):
        return True

    return False


def filter_examples(
    formality_regex: str, example: dict[str, str | int | dict[str, str]]
) -> bool:
    """Filter for examples with enough context and one formality word"""
    if (
        len(re.findall(formality_regex, example['translation']['nl'], re.IGNORECASE))
        != 1
    ):
        return False

    # check if the example has three context sentences
    if len(example['context']['nl']) != 3:
        return False

    return True


def prompt_examples(
    formality_regex: str, dataset: Dataset, examples_needed: int | None
) -> Dataset:
    """Ask the user if formality is clear only from the context sentences."""

    # change examples needed if not provided (so all examples will be prompted)
    examples_needed = len(dataset) + 1 if not examples_needed else examples_needed

    good_examples_dataset = Dataset.from_dict({})

    for idx, example in enumerate(dataset):
        # stop if enough examples are found
        if len(good_examples_dataset) >= examples_needed:
            break

        # prompt the user to verify whether the formality is clear from the context
        print(f'Example {idx + 1}/{len(dataset) + 1}\n')
        newline = '\n'
        print(f'context:\n{newline.join(example["context"]["nl"])}')
        match = re.search(formality_regex, example['translation']['nl'], re.IGNORECASE)
        formality_underlined = f"\033[4m{match.group()}\033[0m"
        print(
            '\nsentence:\n'
            + example['translation']['nl'][: match.span()[0]]
            + formality_underlined
            + example['translation']['nl'][match.span()[1] :],
        )
        print()

        input_ = ''
        while input_ not in ['y', 'n', 'c']:
            input_ = (
                input('Is the formality clear from the context? (y/n): ')
                .strip()
                .lower()
            )

        match input_:
            case 'y':
                good_examples_dataset = good_examples_dataset.add_item(example)
                print(
                    f'\nExample added to the good examples dataset: {len(good_examples_dataset)} / {examples_needed}\n'
                )
            case 'n':
                print()
                pass
            case 'c':
                print('Exiting the program.')
                break

        print('---' * 20)

    return good_examples_dataset


def main(argv: list[str]) -> None:
    """Filter the IWSLT 2017 Dutch-English dataset for formality."""

    # load the dataset
    dataset = load_dataset(
        'gsarti/iwslt2017_context', 'iwslt2017-en-nl', split='train'
    )  # change to train after testing
    # print(f'Total amount of rows in train set: {len(dataset)}')

    # add context fields to the dataset
    dataset = add_context_sentences(dataset)

    # filter the dataset for formality
    # formality_words_dutch = ['u', 'je', 'jij', 'jou', 'jouw', 'uw', 'jullie']
    formality_words = ['u', 'jij', 'jou', 'jouw', 'uw', 'jullie']
    word_boundary = r'\b'
    formality_regex = fr'{"|".join([word_boundary + word + word_boundary for word in formality_words])}'

    # filter the dataset on formality examples
    dataset = dataset.filter(lambda x: find_formality_dutch(formality_regex, x))

    # remove sentences without enough context or with multiple formality words
    dataset = dataset.filter(lambda x: filter_examples(formality_regex, x))

    # shuffle the dataset
    shuffled_dataset = dataset.shuffle(seed=123)

    # prompt the user for correct examples (where formality is decided my context)
    good_examples = prompt_examples(formality_regex, shuffled_dataset, 10)

    # save the final dataset to disk
    save_path = 'good_examples'
    if len(argv) > 1:
        save_path = argv[1]

    if len(good_examples) != 0:
        good_examples.save_to_disk(f'{save_path}/')
        good_examples.to_csv(f'{save_path}.csv')

    print(f'Total amount of good examples saved to {save_path}: {len(good_examples)}')


if __name__ == '__main__':
    main(sys.argv)
