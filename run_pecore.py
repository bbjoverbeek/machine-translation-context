from datasets import load_from_disk
import os
# inseq seems to only work when doing 'pip install -r requirements.txt'
import inseq
from inseq.commands.attribute_context.attribute_context import attribute_context_with_model, AttributeContextArgs

data = load_from_disk("./annotations/combined_and_sorted")

print(f'columns: {data.column_names}')
print(f'nrows: {data.num_rows}')

languages_to_run = ['de', 'nl', 'ru']

# name of the MBART model 'tgt_lang' language identifier
language_to_tgt_lang = {'de': 'de_DE', 'nl': 'nl_XX', 'ru': 'ru_RU'}

# create output folders
for _, folder in language_to_tgt_lang.items():
    if not os.path.isdir(f'./output/{folder}'):
        os.mkdir(f'./output/{folder}')


def get_pecore_args(
        target_language,
        input_context_text,
        input_current_text,
        output_context_text,
        sample_index
):
    return AttributeContextArgs(
        model_name_or_path="facebook/mbart-large-50-one-to-many-mmt",
        attribution_method="saliency",
        attributed_fn="contrast_prob_diff",
        context_sensitivity_metric="kl_divergence",
        context_sensitivity_std_threshold=0,
        attribution_std_threshold=2,
        attribution_topk=5,
        input_context_text=input_context_text,
        input_current_text=input_current_text,
        output_context_text=output_context_text,
        # output_current_text=output_current_text, # used to force a specific output
        contextless_input_current_text="""{current}""",
        input_template="""{context} {current}""",
        contextless_output_current_text="""{current}""",
        output_template="{context} {current}",
        save_path=f"output/{target_language}/{sample_index}.json",
        viz_path=f"output/{target_language}/{sample_index}.html",
        tokenizer_kwargs={'src_lang': 'en_XX', 'tgt_lang': target_language},
    )


def get_pecore_args_for_row(language_id, row, sample_index):
    return get_pecore_args(
        language_to_tgt_lang[language_id],
        row['concatenated_context']['en'],
        row['translation']['en'],
        row['concatenated_context'][language_id],
        sample_index=sample_index
    )


for language in languages_to_run:

    inseq_model = inseq.load_model(
        "facebook/mbart-large-50-one-to-many-mmt",
        "saliency",
        tokenizer_kwargs={'src_lang': 'en_XX', 'tgt_lang': language_to_tgt_lang[language]},
    )

    for i, row in enumerate(data):
        try:
            pecore_args = get_pecore_args_for_row(language, row, i)
            out = attribute_context_with_model(pecore_args, inseq_model)
        except ValueError as e:
            print()
            print(f"ERROR for row {i} with language [{language}]")
            print(e)
            print()
