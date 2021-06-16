"""Class for mulitlingual NER wiki dataset."""
import datasets


class MutlilingNERWiki():
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
        dataset_hi = self._load_dataset_hi()

        dataset_train = datasets.concatenate_datasets([dataset_pt['train'], dataset_es['train'],
                                                       dataset_en['train'], dataset_hi['train']]).shuffle(seed=self._seed)
        dataset_eval = datasets.concatenate_datasets([dataset_pt['validation'], dataset_es['validation'],
                                                       dataset_en['validation'], dataset_hi['validation']]).shuffle(seed=self._seed)
        dataset_test = datasets.concatenate_datasets([dataset_pt['test'], dataset_es['test'],
                                                       dataset_en['test'], dataset_hi['test']]).shuffle(seed=self._seed)

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
    
    def load_dataset_hi(self):
        """Load Hindi part of the dataset.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        dataset_hi = self._load_dataset_pt()
        return (dataset_hi['train']['tokens'], dataset_hi['train']['ner_tags'], dataset_hi['validation']['tokens'],
                dataset_hi['validation']['ner_tags'], dataset_hi['test']['tokens'], dataset_hi['test']['ner_tags'])


    def _load_dataset_en(self):
        """Load English part of the dataset.

        Returns:
            datasets.DatasetDict: the English dataset.
        """
        dataset_en = datasets.load_dataset('wikiann', 'en')
        dataset_en.remove_columns_('langs')
        dataset_en = dataset_en.map(self._renaming_ner_classes_removing_unwanted_classes)
        return dataset_en

    def _load_dataset_es(self):
        """Load Spanish part of the dataset.

        Returns:
            datasets.DatasetDict: the Spanish dataset.
        """
        dataset_es = datasets.load_dataset('wikiann', 'es')
        dataset_es.remove_columns_('langs')
        dataset_es = dataset_es.map(self._renaming_ner_classes_removing_unwanted_classes)
        return dataset_es

    def _load_dataset_pt(self):
        """Load Portuguese part of the dataset.

        Returns:
            datasets.DatasetDict: the Portuguese dataset.
        """
        dataset_pt = datasets.load_dataset('wikiann', 'pt')
        dataset_pt.remove_columns_('langs')
        dataset_pt = dataset_pt.map(self._renaming_ner_classes_removing_unwanted_classes)
        return dataset_pt

    def _load_dataset_hi(self):
        """Load Hindi part of the dataset.

        Returns:
            datasets.DatasetDict: the Hindi dataset.
        """
        dataset_hi = datasets.load_dataset('wikiann', 'hi')
        dataset_hi.remove_columns_('langs')
        dataset_hi = dataset_hi.map(self._renaming_ner_classes_removing_unwanted_classes)
        return dataset_hi

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
