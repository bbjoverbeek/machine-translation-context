# Formal Language: Understanding Context Usage in Machine Translation

## Abstract

Despite the efficiency of machine translation (MT) models, to trust them for problem-solving tasks, it's essential to make sure that their decisions are grounded in linguistic validity. The recent research of Sarti and colleagues (2024) revealed that examining the inner workings of MT models can unveil how they link context to predictions on the example of the gender choice in French. In this paper, we expand previous analysis to identify the uses of context and its validity (from the native speakers' perspectives) in the MT model in the case of formal language. Our work covers 3 pairs of languages: English to Dutch, Russian, and German. By implementing the PECoRe framework (Sarti et al., 2024), identifying both the context-sensitive words and their cues in the context, we explore 20 handcrafted samples of interest in each language pair. Our findings show that model not only often look at the "wrong" context when deciding about formal language words but also more frequently pays attention to non-target tokens. The present work emphasizes that models could lack linguistic validity and future research is needed in order to constrain the model's attention to a certain phenomenon or expand the data of linguistic interest.

## Annotation

An object in Dataset format:

```
doc_id : document ID from the source iwslt2017_context dataset
seg_id : segment ID from the source iwslt2017_context dataset
translation : target sentence in English to translate
context : context sentences in English and one of three languages: Dutch/Russian/German
```

## Usage instructions

Clone the repository

```
git clone https://github.com/bbjoverbeek/machine-translation-context.git
```

You should run all scripts from the root directory

```
cd machine-translation-context
```

Create a virtual environment and install the dependencies

```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

_Note: the Inseq version required to run this project might not be available via pip. If you run into errors, try installing it straight from GitHub:_ `pip install git+https://github.com/inseq-team/inseq.git`

### Find formality

To run the pre-processing of the iwslt2017_context dataset and prompt the user to annotate good examples using context to determine formality, run the following script:

```
python find_formality.py
```

_Optionally you can provide a path to save the annotated examples to. This path defaults to \`good_examples\`._

This will pre-process the data and prompt the user unitil they have provided 10 examples or returned early by providing the 'c' option in the prompt.

### Run the translation script:

Parameters:

- lang: the language to translate the context into
- save_path: the path to save created dataset

```
python translate.py --lang your_lang --save_path your_save_path
```

### PECoRe: context-sensitive words and their context detection:
The `run_pecore.py` script runs PECoRe analysis on a dataset and stores the results for each language in the `output` folder. It is configured to be used on the `combined_and_sorted` dataset, which includes the English target sentences and contexts in Dutch, English, German, and Russian.

If one would like to add a new language, the `combine_datasets.py` script can be adapted to include the new language after following the translation instructions.


### Dataset statistics

The statistics mentioned in the paper about the dataset can be found by running the code in the [dataset_exploration notebook](./dataset_exploration.ipynb). For ease of use the output has been saved.

## Useful links

- [Project page](https://sites.google.com/rug.nl/ik-nlp-2024/projects-description/understanding-context-usage-in-machine-translation?authuser=0)
- [PECoRe paper](https://arxiv.org/abs/2310.01188)
