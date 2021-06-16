"""DataLoader class.

Create a class to load the data files from the shared tasks.
"""
import json
from pathlib import Path
import re

from event_extraction.data_loader.loader import Loader


class ProtestNews2019():
    """DataLoader class.

    Load the data from ProtestNews2019.
    """

    def __init__(self, data_folder="data"):
        """Init function for DataLoader.

        Args:
            data_folder (str, optional): folder with the data of the shared
                task. Defaults to "data".
        """
        self._data_folder = data_folder
        self._loader = Loader()

    def load_task1(self):
        """Load the first task, document classification.

        Returns:
            dict: dictionary with the texts and labels of the dataset
        """
        train_texts, train_labels = self.load_train_task1()
        eval_texts, eval_labels = self.load_eval_task1()
        test_texts, test_labels = self.load_test_task1()
        test_china_texts, test_china_labels = self.load_test_china_task1()
        data_dict = {"train_texts": train_texts,
                "train_labels": train_labels,
                "eval_texts": eval_texts,
                "eval_labels": eval_labels,
                "test_texts": test_texts,
                "test_labels": test_labels,
                "test_china_texts": test_china_texts,
                "test_china_labels": test_china_labels}

        return data_dict

    def load_task3(self):
        """Load the third task, document classification.

        Returns:
            dict: dictionary with the texts and labels of the dataset
        """
        train_texts, train_labels = self.load_train_task3()
        eval_texts, eval_labels = self.load_eval_task3()
        test_texts, test_labels = self.load_test_task3()
        test_china_texts, test_china_labels = self.load_test_china_task3()

        data_dict = {"train_texts": train_texts,
                     "train_labels": train_labels,
                     "eval_texts": eval_texts,
                     "eval_labels": eval_labels,
                     "test_texts": test_texts,
                     "test_labels": test_labels,
                     "test_china_texts": test_china_texts,
                     "test_china_labels": test_china_labels}

        return data_dict

    def load_train_task1(self):
        """Load the train part of the first task, document classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder) / 'Document' / 'train_filled.json'
        data_text, data_label = self._loader.read_json(data_path)
        return (data_text, data_label)


    def load_train_task3(self):
        """Load the train part of the third task, document classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder) / 'task3public9may' / 'train.txt'
        data_text, data_label = self._loader.read_token_annotated_file(data_path)
        return (data_text, data_label)

    def load_eval_task1(self):
        """Load the eval part of the first task, document classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder) / 'Document' / 'dev_filled.json'
        data_text, data_label = self._loader.read_json(data_path)
        return (data_text, data_label)

    def load_eval_task3(self):
        """Load the eval part of the third task, sequence classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder) / 'task3public9may' / 'dev.txt'
        data_text, data_label = self._loader.read_token_annotated_file(data_path)
        return (data_text, data_label)


    def load_test_task1(self):
        """Load the test part of the first task, document classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder) / 'Document' / 'test_filled.json'
        data_text, data_label = self._loader.read_json(data_path)
        return (data_text, data_label)

    def load_test_task3(self):
        """Load the test part of the third task, sequence classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder) / 'task3public9may' / 'task3_test.data'
        data_text, data_label = self._loader.read_token_annotated_file(data_path)
        return (data_text, data_label)

    def load_test_china_task1(self):
        """Load the train part of the first task, document classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder)
        data_path = data_path / 'Document' / 'test_china_filled.json'
        data_text, data_label = self._loader.read_json(data_path)
        return (data_text, data_label)

    def load_test_china_task3(self):
        """Load the train part of the first task, document classification.

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        data_path = Path(self._data_folder)
        data_path = data_path / 'task3public9may' / 'china_test.data'
        data_text, data_label = self._loader.read_token_annotated_file(data_path)
        return (data_text, data_label)
