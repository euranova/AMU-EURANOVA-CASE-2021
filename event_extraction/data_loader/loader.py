"""General loader class."""

import json
from pathlib import Path
import re

class Loader():
    """General loader class."""

    def read_token_annotated_file(self, file_path):
        """Read a token annotated file, i.e. a document with each token on a line with or without a tab-separated annotation.

        Each token is on a line, and if there are annotations they are after the word separated with a tab.

        Args:
            file_path (str): path of the document 

        Returns:
            list of list of str, list of list of str: list of the docs and list of the BIO tags
        """
        file_path = Path(file_path)

        raw_text = file_path.read_text().strip()
        raw_docs = re.split(r'\n\t?\n', raw_text)
        token_docs = []
        tag_docs = []
        for doc in raw_docs:
            tokens = []
            tags = []
            for line in doc.split('\n'):
                if '\t' in line:
                    token, tag = line.split('\t')
                    tokens.append(token)
                    tags.append(tag)
                else:
                    tokens.append(line)
            token_docs.append(tokens)
            if tags != []:
                tag_docs.append(tags)

        if tag_docs == [] or (len(tag_docs) != len(token_docs)) or\
            not all([len(tag_doc)==len(text_doc) for tag_doc, text_doc in zip (tag_docs, token_docs)]):
            tag_docs = None

        return token_docs, tag_docs

    def read_json(self, data_path, txt_name='text'):
        """Extract data from json formatted file.

        Args:
            data_path (Path): path of the file with data

        Returns:
            (list of str, list of int): output list of texts and the list of
                their labels.
        """
        with open(data_path, "r") as data_file:
            data = data_file.read()
        data_json = [
                json.loads(elem) for elem in data.split('\n') if elem != ""
            ]
        data_text = [elem[txt_name] for elem in data_json]

        if 'label' in data_json[0].keys():
            data_label = [elem['label'] for elem in data_json]
        else:
            data_label = []
        return (data_text, data_label)
