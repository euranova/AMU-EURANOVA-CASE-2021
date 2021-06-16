"""Generate general statistics on a dataset for a sequence task under a BIO format."""


from collections import Counter
from pathlib import Path

import plotly.express as px

class SequenceDataset():
    """The goal of this class is to generate general statistics on a BIO-formatted dataset."""

    def __init__(self, results_folder='results'):
        """Init function of the class.

        Args:
            results_folder (str, optional): Place where the results will be. Defaults to 'results'.
        """
        self._result_folder = Path(results_folder)

    def generate_stats(self, texts, labels, dataset_name):
        """Global function generating all the stats.

        Args:
            texts (list str): texts of the dataset
            labels (list list str): labels of the dataset
            dataset_name(str): name of the dataset
        """
        print_folder = self._result_folder / dataset_name
        print_folder.mkdir(parents=True, exist_ok=True)

        # First, we gather all the data we want to save.
        avg_len_texts = self._average_len_texts(texts)
        histogram_len_texts = self._histogram_len_text(texts)
        labels_list = self._generate_list_labels(labels)
        nb_doc_labels, fig_nb_doc_labels = self._number_documents_with_label(labels)
        avg_app_labels, fig_avg_app_labels = self._avg_app_label(labels)
        avg_len_labels, fig_avg_len_labels = self._avg_len_label(labels)

        results_file_path = print_folder / 'general_statistics.txt'

        with results_file_path.open("w", encoding="utf-8") as results_file:
            results_file.write('Average length of texts: {}\n'.format(avg_len_texts))
            results_file.write('Label lists: {}\n'.format(labels_list))
            results_file.write('Number of docs with at least one label: {}\n'.format(nb_doc_labels))
            results_file.write('Average number of a specific label in a document it appears: {}\n'.format(avg_app_labels))
            results_file.write('Average length of a specific label: {}\n'.format(avg_len_labels))

        histogram_len_texts.write_html(str(print_folder / 'histogram_len_texts.html'))
        fig_nb_doc_labels.write_html(str(print_folder / 'nb_docs_with_label.html'))
        fig_avg_app_labels.write_html(str(print_folder / 'avg_nb_labels.html'))
        fig_avg_len_labels.write_html(str(print_folder / 'avg_len_labels.html'))

    def _average_len_texts(self, texts):
        """Compute the average lengths of the texts in a dataset.

        Args:
            texts (list(str)): Texts of the dataset

        Returns:
            int: Average length of the texts.
        """
        len_texts = [len(text) for text in texts]

        avg_len = sum(len_texts) / len(len_texts)

        return avg_len

    def _histogram_len_text(self, texts):
        """Generate histogram of texts length.

        Args:
            texts (list str): texts of the dataset

        Returns:
            plotly.graph_objs._figure.Figure: histogram of the texts length
        """
        len_texts = [len(text) for text in texts]

        fig = px.histogram(x=len_texts, labels={'x':'Number of words in texts'},
                   title='Number of documents in function of their length')
        return fig

    def _generate_list_labels(self, labels):
        """Generate the list of the label existing in the dataset.

        Args:
            labels (list list str): labels of the texts in the dataset.

        Returns:
            list str: list of the all the labels exisiting in the dataset
        """
        count_labels = [Counter(doc_labels) for doc_labels in labels]

        list_labels = []

        _ = [
            [list_labels.append(label) for label in list(doc_labels.keys()) if (label not in list_labels)]
            for doc_labels in count_labels
        ]

        return list_labels

    def _generate_nb_doc_with_label(self, labels):
        """Generate dict with number of documents with at least one label for each label.

        Args:
            labels (list list str): labels of the texts in the dataset.

        Returns:
            dict: label and number of documents with this label
        """
        count_labels = [Counter(labels) for labels in labels]
        list_labels = self._generate_list_labels(labels)
        
        dict_nb_doc_with_label = {label:0 for label in list_labels if label[0]!='I'}

        for doc in count_labels:
            for label in list_labels:
                if label in list(doc.keys()) and label[0] != 'I':
                    dict_nb_doc_with_label[label] += 1

        return dict_nb_doc_with_label

    def _generate_nb_labels_in_dataset(self, labels):
        """Generate dict with number of label in dataset for each labels.

        Args:
            labels (list list str): labels of the text in the dataset.

        Returns:
            dict: for each label the number of times they appear.
        """
        count_labels = [Counter(labels) for labels in labels]
        list_labels = self._generate_list_labels(labels)

        dict_nb_labels_in_dataset = {label:0 for label in list_labels}

        for doc in count_labels:
            for label, nb_app in doc.items():
                dict_nb_labels_in_dataset[label] += nb_app

        return dict_nb_labels_in_dataset

    def _number_documents_with_label(self, labels):
        """Generate data and graph about number of documents with a label.

        Args:
            labels (list list str): labels of the texts in the dataset.

        Returns:
            dict str, plotly.graph_objs._figure.Figure: data in dict, figure representing it.
        """        
        dict_nb_doc_with_label = self._generate_nb_doc_with_label(labels)
        
        fig = px.bar(x=dict_nb_doc_with_label.keys(), y=dict_nb_doc_with_label.values(), 
                     labels={'x': 'Label', 'y': 'Number of document with at least 1 label'},
                     title='Number of documents with at least one annotation for each annotation type')

        return dict_nb_doc_with_label, fig

    def _avg_app_label(self, labels):
        """Generate data and figure to represent the average apparition of each label in a document.

        Note: we only count the documents where the label appears at least once.

        Args:
            labels (list list str): labels of the texts in the dataset.

        Returns:
             dict str, plotly.graph_objs._figure.Figure: data in dict, figure representing it.
        """
        dict_nb_doc_with_label = self._generate_nb_doc_with_label(labels)
        dict_nb_labels_in_dataset = self._generate_nb_labels_in_dataset(labels)

        # We only want to count the B tags and not the I.
        dict_nb_start_labels = {label:nb_app for label, nb_app in dict_nb_labels_in_dataset.items() if label[0]=='B'}

        # We only want to count the documents where the label appear at least once.
        dict_nb_avg_app_label = {
            label:(dict_nb_start_labels[label]/dict_nb_doc_with_label[label]) for label in dict_nb_start_labels.keys()
        }

        fig = px.bar(x=dict_nb_avg_app_label.keys(), y=dict_nb_avg_app_label.values(), 
                     labels={'x': 'Label', 'y': 'Average number of apparition'},
                     title='Average number of each annotation type by document when they appear')

        return dict_nb_avg_app_label, fig

    def _avg_len_label(self, labels):
        """Generate data and figure to represent the average length of each label type.

        Args:
            labels (list list str): labels of the texts in the dataset.

        Returns:
             dict str, plotly.graph_objs._figure.Figure: data in dict, figure representing it.
        """
        dict_nb_doc_with_label = self._generate_nb_doc_with_label(labels)
        dict_nb_labels_in_dataset = self._generate_nb_labels_in_dataset(labels)

        # We only want to count the B tags and not the I.
        dict_nb_start_labels = {label:nb_app for label, nb_app in dict_nb_labels_in_dataset.items() if label[0]=='B'}

        # To compute the average we just need to sum the total length of each label 
        # and then to divide by the number of labels.
        dict_avg_len_label = {
            label:(
                (dict_nb_labels_in_dataset[label] + dict_nb_labels_in_dataset['I'+label[1:]])
                /dict_nb_labels_in_dataset[label])
            for label in dict_nb_start_labels.keys()
        }

        fig = px.bar(x=dict_avg_len_label.keys(), y=dict_avg_len_label.values(), 
                     labels={'x': 'Label', 'y': 'Number of tokens'},
                     title='Average length of each annotation for each type')

        return dict_avg_len_label, fig


    


    