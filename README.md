# Formal Language: Understanding Context Usage in Machine Translation

[Project page](https://sites.google.com/rug.nl/ik-nlp-2024/projects-description/understanding-context-usage-in-machine-translation?authuser=0) describing the task.

## Abstract
Despite the efficiency of machine translation (MT) models, to trust them for problem-solving tasks, it's essential to make sure that their decisions are grounded in linguistic validity. The recent research of Sarti and colleagues (2024) revealed that examining the inner workings of MT models can unveil how they link context to predictions on the example of the gender choice in French. In this paper, we expand previous analysis to identify the uses of context and its validity\footnote{from the native speakers' perspectives} in the MT model in the case of formal language. Our work covers 3 pairs of languages: English to Dutch, Russian, and German. By implementing the PECoRe framework (Sarti et al., 2024), identifying both the context-sensitive words and their cues in the context, we explore 20 handcrafted samples of interest in each language pair. Our findings show that model not only often look at the "wrong" context when deciding about formal language words but also more frequently pays attention to non-target tokens. The present work emphasizes that models could lack linguistic validity and future research is needed in order to constrain the model's attention to a certain phenomenon or expand the data of linguistic interest.

## Annotation example
An object in Dataset format:
```
- doc_id : document ID from the source *iwslt2017_context* dataset
- seg_id : segment ID from the source *iwslt2017_context* dataset
- translation : target sentence in English to translate
- context : context sentences in English and one of three languages: Dutch/Russian/German
```

## Code usage instructions
