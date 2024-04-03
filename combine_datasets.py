import datasets

data_nl = datasets.load_from_disk("./annotations/annotations_nl")
data_ru = datasets.load_from_disk("./annotations/annotations_ru")
data_de = datasets.load_from_disk("./annotations/annotations_de")

data_nl = data_nl.sort(['doc_id', 'seg_id'])
data_ru = data_ru.sort(['doc_id', 'seg_id'])
data_de = data_de.sort(['doc_id', 'seg_id'])

combined_set = []
for nl, ru, de in zip(data_nl, data_ru, data_de):
    new_item = {}
    assert (nl['translation']['en'] == de['translation']['en'] == ru['translation']['en'])
    new_item['doc_id'] = nl['doc_id']
    new_item['seg_id'] = nl['seg_id']

    new_item['translation'] = {}
    new_item['translation']['en'] = nl['translation']['en']

    new_item['context'] = {}
    new_item['context']['en'] = nl['context']['en']
    new_item['context']['nl'] = nl['context']['nl']
    new_item['context']['de'] = de['context']['de']
    new_item['context']['ru'] = ru['context']['ru']

    new_item['concatenated_context'] = {
        lan: ' '.join(context) for lan, context in new_item['context'].items()
    }


    def do_clean(string):
        """# something went wrong with the data, do a cleanup of '[' and ']' and double spaces in all strings"""
        return string.replace('[', '').replace(']', '').replace('  ', ' ')


    for k, v in new_item.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                if isinstance(v2, str):
                    v[k2] = do_clean(v2)
                elif isinstance(v2, list):
                    v[k2] = [do_clean(s) for s in v2]
        elif isinstance(v, str):
            new_item[k] = do_clean(v)

    combined_set.append(new_item)

new_set = datasets.Dataset.from_list(combined_set)

new_set.save_to_disk('annotations/combined_and_sorted')
new_set.to_csv('annotations/combined_and_sorted.csv')
