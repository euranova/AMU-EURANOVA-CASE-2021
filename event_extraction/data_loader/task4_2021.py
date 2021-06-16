"""Class to load task 4 form 2021 shared task."""
import numpy as np
from sklearn.model_selection import train_test_split

from event_extraction.data_loader.loader import Loader

class Task4_2021():
    """Load task 4 from 2021 shared task."""

    def __init__(self, path_data="data/task_2021/subatask4-token", seed=13, data_order_seed=13):
        """Init function for load task4.

        Args:
            path_data (str, optional): path to datafiles. Defaults to "data/task_2021/subatask4-token".
        """
        self._path_data = path_data
        self._seed = seed
        self._data_order_seed = data_order_seed
        self._loader = Loader()
        self._texts_en, self._labels_en = self._loader.read_token_annotated_file(self._path_data / "en-train.txt")
        self._texts_es, self._labels_es = self._loader.read_token_annotated_file(self._path_data / "es-train.txt")
        self._texts_pt, self._labels_pt = self._loader.read_token_annotated_file(self._path_data / "pr-train.txt")

    def load_dataset(self, eval_size=0.2, test_size=0.2):
        """Load dataset with all the languages.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        # We want all the splits to contain the same amount of texts in each language
        train_texts_en, train_labels_en, eval_texts_en, eval_labels_en, test_texts_en, test_labels_en = \
            self.load_dataset_en(eval_size, test_size)

        train_texts_es, train_labels_es, eval_texts_es, eval_labels_es, test_texts_es, test_labels_es = \
            self.load_dataset_es(eval_size, test_size)

        train_texts_pt, train_labels_pt, eval_texts_pt, eval_labels_pt, test_texts_pt, test_labels_pt = \
            self.load_dataset_pt(eval_size, test_size)

        # We now create our splits with all the languages
        train_texts = train_texts_en + train_texts_es + train_texts_pt
        eval_texts = eval_texts_en + eval_texts_es + eval_texts_pt
        test_texts = test_texts_en + test_texts_es + test_texts_pt
        train_labels = train_labels_en + train_labels_es + train_labels_pt
        eval_labels = eval_labels_en + eval_labels_es + eval_labels_pt
        test_labels = test_labels_en + test_labels_es + test_labels_pt

        # We shuffle the list and labels to shuffle the languages between them
        rng = np.random.RandomState(seed=self._data_order_seed)
        train_texts, train_labels = self._shuffle_texts_labels(train_texts, train_labels, rng)
        eval_texts, eval_labels = self._shuffle_texts_labels(eval_texts, eval_labels, rng)
        test_texts, test_labels = self._shuffle_texts_labels(test_texts, test_labels, rng)

        return train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels

    def load_dataset_en(self, eval_size=0.2, test_size=0.2):
        """Load English part of the dataset.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        train_texts_en, train_labels_en, eval_texts_en, eval_labels_en, test_texts_en, test_labels_en = \
            self._train_eval_test_split(self._texts_en, self._labels_en, eval_size=eval_size, test_size=test_size, random_state=self._seed)
        return train_texts_en, train_labels_en, eval_texts_en, eval_labels_en, test_texts_en, test_labels_en

    def load_dataset_es(self, eval_size=0.2, test_size=0.2):
        """Load Spanish part of the dataset.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        train_texts_es, train_labels_es, eval_texts_es, eval_labels_es, test_texts_es, test_labels_es = \
            self._train_eval_test_split(self._texts_es, self._labels_es, eval_size=eval_size, test_size=test_size, random_state=self._seed)
        return train_texts_es, train_labels_es, eval_texts_es, eval_labels_es, test_texts_es, test_labels_es

    def load_dataset_pt(self, eval_size=0.2, test_size=0.2):
        """Load Portuguese part of the dataset.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        train_texts_pt, train_labels_pt, eval_texts_pt, eval_labels_pt, test_texts_pt, test_labels_pt = \
            self._train_eval_test_split(self._texts_pt, self._labels_pt, eval_size=eval_size, test_size=test_size, random_state=self._seed)
        return train_texts_pt, train_labels_pt, eval_texts_pt, eval_labels_pt, test_texts_pt, test_labels_pt

    def _train_eval_test_split(self, texts, labels, eval_size, test_size, random_state):
        """Split documents into train, eval and test.

        Args:
            texts (list): list of texts.
            labels (list): list of labels.
            eval_size (float): percentage of the full set into eval.
            test_size (float): percentage of the full set into test.
            random_state (int): seed for random state.

        Returns:
            list, list, list, list, list, list: train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels
        """
        if eval_size+test_size != 0:
            train_texts, eval_test_texts, train_labels, eval_test_labels = \
                train_test_split(texts, labels, test_size=eval_size+test_size, random_state=13)
                
            if test_size != 0:
                eval_texts, test_texts, eval_labels, test_labels = \
                    train_test_split(eval_test_texts, eval_test_labels, test_size=test_size/(eval_size+test_size), random_state=13)
            else:
                eval_texts = eval_test_texts
                eval_labels = eval_test_labels
                test_texts, test_labels = [], []

        else:
            train_texts = texts
            train_labels = labels
            eval_texts, test_texts = [], []
            eval_labels, test_labels = [], []
            
        return train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels

    def _shuffle_texts_labels(self, texts, labels, random_generator):
        """Shuffle texts and labels into a new order while keepoing the texts-labels correspondance.

        Args:
            texts (list): texts to sort.
            labels (list): labels to sort.
            random_generator (np.random.RandomState): the random generator.

        Returns:
            texts, labels: randomly sorted texts, labels.
        """
        if texts != []:
            data = list(zip(texts, labels))
            random_generator.shuffle(data)

            texts, labels = list(zip(*data))
            shuffled_texts = list(texts)
            shuffled_labels = list(labels)
        else:
            shuffled_texts, shuffled_labels = [], [] 
        return shuffled_texts, shuffled_labels