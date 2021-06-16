"""BERT Classifier for sentence/document classification."""
import logging
import os

import mlflow
import numpy as np
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from event_extraction.model.torch_dataset import TorchDataset

class BertSequenceClassif():
    """Class to finetune BERT to a specific dataset of sequence classification."""

    def __init__(self, pretrained_model, training_arguments):
        """Inititalisation of the model.

        Args:
            pretrained_model (str): path or name of pretrained model
            training_arguments (transformers.TrainingArguments) : arguments to train the model
        """
        self._training_args = training_arguments
        self._pretrained_model = pretrained_model
        self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model)
        self._model = AutoModelForSequenceClassification.from_pretrained(self._pretrained_model)
        self._trainer = None
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train(self, train_texts, train_labels, eval_texts, eval_labels):
        """Training of our model.

        Args:
            train_texts (list of str): Texts of the training part
            train_labels (lst of int): Labels of the training part
            eval_texts (list of str): Texts of the eval part
            eval_labels (list of int): Labels of the eval part
        """
        # We tokenize our texts to use them in BERT
        train_encodings = self._tokenize_texts(train_texts)
        eval_encodings = self._tokenize_texts(eval_texts)

        train_labels_encoded = self._encode_labels(train_labels, train_encodings)
        eval_labels_encoded = self._encode_labels(eval_labels, eval_encodings)

        # As we use Torch as backend, the data is transformed in a Dataset object
        train_dataset = TorchDataset(train_encodings, train_labels_encoded)
        eval_dataset = TorchDataset(eval_encodings, eval_labels_encoded)

        self._model.resize_token_embeddings(len(self._tokenizer))

        # We can now specify the model
        self._trainer = Trainer(
            model=self._model,
            args=self._training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self._compute_metrics,
        )

        self._trainer.train()
        mlflow.log_param("loss_name", "default")

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        f1_score_value = f1_score(labels, predictions)

        return {"f1": f1_score_value}

    def save_model(self, output_dir=None):
        """Save the model, can be loaded by load_from_pretrained.

        Args:
            output_dir (str, optional): path to save the model. Defaults to None.
        """
        self._trainer.save_model(output_dir)

    def predict(self, test_texts, test_labels=None):
        """Predict the results of the trained model.

        Specifically, the labels are not be compulsory. It returns
        the predictions and the score if the labels are available.

        Args:
            test_texts (list of str): Texts of the test part
            test_labels (list of int): Labels of the test part
        Returns:
            NamedTuple: score in function of metrics
        """
        if not self._trainer :
            logging.error('The model has not been trained yet.')
        else:
            # We tokenize our texts to use them in BERT
            test_encodings = self._tokenize_texts(test_texts)
            test_labels_available = True
            results = None

            if not test_labels:
                test_labels = [0]*len(test_texts)
                test_labels_available = False

            test_labels_encoded = self._encode_labels(test_labels, test_encodings)

            test_dataset = TorchDataset(test_encodings, test_labels_encoded)

            predictions, labels, _ = self._trainer.predict(test_dataset)
            
            # Create an empty list for each doc.
            pred = [[] for _ in test_texts]

            # Add all the predictions for a specific doc into it list.
            for i, doc_id in enumerate(test_encodings['overflow_to_sample_mapping']):
                pred[doc_id].append(predictions[i])

            # Mean on all the predictions of one document.
            decoded_predictions = [np.mean(score_pred, axis=0) for score_pred in pred]

            # Argmax to find the predicted class.
            predictions = np.argmax(decoded_predictions, axis=1)

            if test_labels_available:
                results = {"accuracy": (np.array(predictions)==np.array(test_labels)).mean()}

            return results, predictions

    def _tokenize_texts(self, texts):
        """Tokenize the texts.

        Args:
            texts (str): pre-tokenized texts with each words an element of a list

        Returns:
            Encoding: the tokenized texts
        """
        tokenized_texts = self._tokenizer(texts, padding="max_length", truncation=True,
                                          return_overflowing_tokens=True, stride=150, max_length=512)
        return tokenized_texts

    def _encode_labels(self, labels, encodings):
        """Encode the labels in case of overflow.

        Args:
            labels (list of int): labels of the docs
            encodings (Encodings): the encodings of the docs

        Returns:
            list of int: labels of the encoded docs (overflow)
        """
        encoded_labels = []
        for doc_id in encodings['overflow_to_sample_mapping']:
            encoded_labels.append(labels[doc_id])

        return encoded_labels


