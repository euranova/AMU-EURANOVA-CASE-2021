"""Class for mulitlingual NER dataset."""
import datasets


class MutlilingNER():
    """Class for mulitlingual NER dataset."""

    def __init__(self, seed=13):
        """Init function of the class."""
        self._seed = seed

    def load_dataset(self):
        """Load the datasets, merge each split of the datasets into one and shuffle all the splits to shuffle the datasets.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        dataset_en = self._load_dataset_en()
        dataset_es = self._load_dataset_es()
        dataset_pt = self._load_dataset_pt()

        dataset_train = datasets.concatenate_datasets([dataset_pt['train'], dataset_es['train'],
                                                       dataset_en['train']]).shuffle(seed=self._seed)
        dataset_eval = datasets.concatenate_datasets([dataset_pt['validation'], dataset_es['validation'],
                                                       dataset_en['validation']]).shuffle(seed=self._seed)
        dataset_test = datasets.concatenate_datasets([dataset_pt['test'], dataset_es['test'],
                                                       dataset_en['test']]).shuffle(seed=self._seed)

        return (dataset_train['tokens'], dataset_train['ner_tags'], dataset_eval['tokens'], dataset_eval['ner_tags'],
                dataset_test['tokens'], dataset_test['ner_tags'])

    def load_dataset_en(self):
        """Load English part of the dataset.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        dataset_en = self._load_dataset_en()
        return (dataset_en['train']['tokens'], dataset_en['train']['ner_tags'], dataset_en['validation']['tokens'],
                dataset_en['validation']['ner_tags'], dataset_en['test']['tokens'], dataset_en['test']['ner_tags'])


    def load_dataset_es(self):
        """Load Spanish part of the dataset.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        dataset_es = self._load_dataset_es()
        return (dataset_es['train']['tokens'], dataset_es['train']['ner_tags'], dataset_es['validation']['tokens'],
                dataset_es['validation']['ner_tags'], dataset_es['test']['tokens'], dataset_es['test']['ner_tags'])

    def load_dataset_pt(self):
        """Load Portuguese part of the dataset.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        dataset_pt = self._load_dataset_pt()
        return (dataset_pt['train']['tokens'], dataset_pt['train']['ner_tags'], dataset_pt['validation']['tokens'],
                dataset_pt['validation']['ner_tags'], dataset_pt['test']['tokens'], dataset_pt['test']['ner_tags'])


    def _load_dataset_en(self):
        """Load English part of the dataset.

        Returns:
            datasets.DatasetDict: the English dataset.
        """
        dataset_en = datasets.load_dataset('conll2003')
        dataset_en.remove_columns_(['pos_tags', 'chunk_tags'])
        dataset_en = dataset_en.map(self._renaming_ner_classes_removing_unwanted_classes)
        return dataset_en

    def _load_dataset_es(self):
        """Load Spanish part of the dataset.

        Returns:
            datasets.DatasetDict: the Spanish dataset.
        """
        dataset_es = datasets.load_dataset('conll2002', 'es')
        dataset_es.remove_columns_('pos_tags')
        dataset_es = dataset_es.map(self._renaming_ner_classes_removing_unwanted_classes)
        return dataset_es

    def _load_dataset_pt(self):
        """Load Portuguese part of the dataset.

        Returns:
            datasets.DatasetDict: the Portuguese dataset.
        """
        dataset_pt = datasets.load_dataset('harem', 'selective')
        dataset_pt = dataset_pt.map(self._renaming_ner_classes_removing_unwanted_classes)
        return dataset_pt

    def _renaming_ner_classes_removing_unwanted_classes(self, example):
        """Rename ner classes and remove unwanted classes.

        Args:
            example (dict): a data from a dataset

        Returns:
            dict: a data with the renaming classes.
        """
        dict_classes = {0:"O", 1:"B-PER", 2:"I-PER", 3:"B-ORG", 4:"I-ORG", 5:"B-LOC", 6:"I-LOC"}
        example['ner_tags'] = [dict_classes[ner_tag] if ner_tag < 7 else "O" for ner_tag in example['ner_tags']]
        return example
