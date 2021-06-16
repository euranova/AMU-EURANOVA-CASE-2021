"""BERT Classifier for token classification."""
import logging
import os
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

from datasets import load_metric
import mlflow
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import HPSearchBackend, BestRun

from event_extraction.model.torch_dataset import TorchDataset
from event_extraction.model.trainer_token_classif import TrainerTokenClassif

IGNORED_TOKEN = -100

class BertTokenClassif():
    """Class to finetune BERT to a specific dataset of sequence classification."""

    def __init__(self, pretrained_model, training_arguments, loss="macro", seed=42, initialization_prediction_layer_seed=42):
        """Inititalisation of the model.

        Args:
            pretrained_model (str): path or name of pretrained model, convention available here:
                https://huggingface.co/transformers/model_doc/auto.html#autotokenizer, if it is a
                path, the folder should contain saved model AND tokenizer (as in save_model)
            training_arguments (transformers.TrainingArguments) : arguments to train the model
            loss (str): either "macro", "micro" or "base"
            seed (int): seed for initialisation of model
            initialization_prediction_layer_seed (int): seed for initilization of prediction layer
        """
        self._training_args = training_arguments
        self._trainer = None
        self._stride = 150
        self._max_length = 512 # We set this number at 512 as BERT-base and many models have this value as max-length
        self._loss = loss
        transformers.set_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        self._seed = seed
        self._initialization_prediction_layer_seed = initialization_prediction_layer_seed

        if Path(pretrained_model).is_dir():
            self._pretrained_model = Path(pretrained_model).absolute()
        else:
            self._pretrained_model = pretrained_model
        self._tokenizer = AutoTokenizer.from_pretrained(self._pretrained_model)

        # Solve a warning, information here: https://stackoverflow.com/questions/62691279/how-to-disable-tokenizers-parallelism-true-false-warning
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def train(self, train_texts, train_labels, dev_texts, dev_labels):
        """Training of our model.

        Args:
            train_texts (list of str): Texts of the training part
            train_labels (lst of int): Labels of the training part
            dev_texts (list of str): Texts of the dev part
            dev_labels (list of int): Labels of the dev part
        """
        train_dataset, dev_dataset = self._preprocess_train_dev_data(train_texts, train_labels, dev_texts, dev_labels)
    
        self._init_trainer(train_dataset, dev_dataset)

        self._trainer.train()

    def hyperparameter_search(self, train_texts, train_labels, dev_texts, dev_labels,
                              hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
                              compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
                              n_trials: int = 20,
                              direction: str = "minimize",
                              backend: Optional[Union["str", HPSearchBackend]] = None,
                              hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
                              **kwargs,
                            ) -> BestRun:
        """Training of our model.

        Taking inspiration from https://docs.ray.io/en/master/tune/examples/pbt_transformers.html

        .. warning::

            To use this method, you need to have provided a ``model_init`` when initializing your
            :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
            with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
            method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.

        Args:
            train_texts (list of str): Texts of the training part
            train_labels (lst of int): Labels of the training part
            dev_texts (list of str): Texts of the dev part
            dev_labels (list of int): Labels of the dev part
            hp_space (:obj:`Callable[["optuna.Trial"], Dict[str, float]]`, `optional`):
                A function that defines the hyperparameter search space. Will default to
                :func:`~transformers.trainer_utils.default_hp_space_optuna` or
                :func:`~transformers.trainer_utils.default_hp_space_ray` depending on your backend.
            compute_objective (:obj:`Callable[[Dict[str, float]], float]`, `optional`):
                A function computing the objective to minimize or maximize from the metrics returned by the
                :obj:`evaluate` method. Will default to :func:`~transformers.trainer_utils.default_compute_objective`.
            n_trials (:obj:`int`, `optional`, defaults to 100):
                The number of trial runs to test.
            direction(:obj:`str`, `optional`, defaults to :obj:`"minimize"`):
                Whether to optimize greater or lower objects. Can be :obj:`"minimize"` or :obj:`"maximize"`, you should
                pick :obj:`"minimize"` when optimizing the validation loss, :obj:`"maximize"` when optimizing one or
                several metrics.
            backend(:obj:`str` or :class:`~transformers.training_utils.HPSearchBackend`, `optional`):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending on which
                one is installed. If both are installed, will default to optuna.
            kwargs:
                Additional keyword arguments passed along to :obj:`optuna.create_study` or :obj:`ray.tune.run`. For
                more information see:

                - the documentation of `optuna.create_study
                  <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html>`__
                - the documentation of `tune.run
                  <https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run>`__

        Returns:
            :class:`transformers.trainer_utils.BestRun`: All the information about the best run.
        """
        train_dataset, dev_dataset = self._preprocess_train_dev_data(train_texts, train_labels, dev_texts, dev_labels)
    
        self._init_trainer(train_dataset, dev_dataset)
        bestrun = self._trainer.hyperparameter_search(hp_space=lambda _:hp_space, compute_objective=compute_objective,
                                                      n_trials=n_trials, direction=direction, backend=backend, hp_name=hp_name,
                                                      **kwargs)
        return bestrun

    def _preprocess_train_dev_data(self, train_texts, train_labels, dev_texts, dev_labels):
        """Preprocess the data.

        Args:
            train_texts (list): list of docs where each doc is a list of words
            train_labels (list): list of docs where each doc is a list of the label of each word
            dev_texts (list): list of docs where each doc is a list of words
            dev_labels (list): list of docs where each doc is a list of the label of each word

        Returns:
            TorchDataset, TorchDataset: dataset object taken as input of a Transformers model
        """
        # We tokenize our texts to use them in BERT
        train_encodings = self._tokenize_texts(train_texts)
        if dev_texts != []:
            dev_encodings = self._tokenize_texts(dev_texts)

        # We need to encode the tags
        self._unique_tags = sorted(list(set(tag for doc in train_labels for tag in doc)))
        self._tag2id = {tag: id for id, tag in enumerate(self._unique_tags)}
        self._id2tag = {id: tag for tag, id in self._tag2id.items()}

        # We want to force the O to 0

        id_O = self._tag2id['O']
        tag_0 = self._id2tag[0]
        self._tag2id[tag_0] =id_O
        self._tag2id['O'] = 0
        self._id2tag[id_O] = tag_0
        self._id2tag[0] = 'O'

        train_labels = self._encode_tags(train_labels, train_encodings)
        if dev_texts != []:
            dev_labels = self._encode_tags(dev_labels, dev_encodings)

        # As we use Torch as backend, the data is transformed in a Dataset object
        train_dataset = TorchDataset(train_encodings, train_labels)
        if dev_texts != []:
            dev_dataset = TorchDataset(dev_encodings, dev_labels)
        else:
            dev_dataset = TorchDataset({"overflow_to_sample_mapping":[]}, [])
        
        return train_dataset, dev_dataset

    def _init_trainer(self, train_dataset, dev_dataset):
        """Init the Trainer class of transformers.

        Args:
            train_dataset (TorchDataset): Pytorch Dataset object for Transformers model
            dev_dataset (TorchDataset): Pytorch Dataset object for Transformers model
        """
        self._metric = load_metric("seqeval", zero_division=0)

        # We can now specify the model
        self._trainer = TrainerTokenClassif(
            model_init=self._get_model,
            args=self._training_args,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            compute_metrics=self._compute_metrics,
            loss=self._loss
        )

    def _get_model(self):
        """Return model. Needed for hyperparameter search.

        Generate a model with a specific seed for the classifier initialization if given in the init.

        Returns:
            transformers.PreTrainedModel: a model.
        """
        # In order to be able to manually set the seed of the classifier we need to cheat a bit.
        # We will first change the size of the bias and the weight to our current needs, esp. needed for
        # behavioral fine-tuning as the task is the same but not the number of classes.
        # And then we will change the classifier with our, which is initialized with a given seed.

        # Reload a model or take on online.
        tmp_model = AutoModelForTokenClassification.from_pretrained(self._pretrained_model)

        # If behavioral fine-tuning, the classifier is the good type but not format, and if we specify a number of classes
        # during the loading it will fail. So, we change the size of the classifier weigths and biases.
        # We also want all the other information about its states that have to be the same.
        state_dict = tmp_model.state_dict().copy()
        state_dict["classifier.bias"] = torch.zeros(len(self._unique_tags))
        state_dict["classifier.weight"] = torch.zeros((len(self._unique_tags), tmp_model.config.hidden_size))

        # We do not need this model anymore.
        del tmp_model

        # Now the classifier have the right dimension as we force them. We just need to initialize the classifier with our seed.
        self._model = AutoModelForTokenClassification.from_pretrained(self._pretrained_model, state_dict=state_dict,
                                                                      num_labels=len(self._unique_tags))
        self._model.resize_token_embeddings(len(self._tokenizer))

        # We set all the seeds for the classifier with a specific value.
        torch.cuda.manual_seed(self._initialization_prediction_layer_seed)
        torch.manual_seed(self._initialization_prediction_layer_seed)
        # We change the classifier, by the same but initialize differently.
        self._model.classifier = nn.Linear(self._model.config.hidden_size, len(self._unique_tags))

        # We then go back to the general random seed.
        torch.cuda.manual_seed(self._seed)
        torch.manual_seed(self._seed)
        return self._model

    def save_model(self, output_dir=None):
        """Save the model, can be loaded by load_from_pretrained.

        Args:
            output_dir (str, optional): path to save the model. Defaults to None.
        """
        self._trainer.save_model(output_dir)
        self._tokenizer.save_pretrained(output_dir)

    def predict(self, test_texts, test_labels=None):
        """Predict the results of the trained model.

        Args:
            test_texts (list of str): Texts of the test part
            test_labels (list of int): Labels of the test part
        Returns:
            NamedTuple, predictions: score in function of metrics, the predictions
        """
        if self._trainer == None :
            logging.info('The model has not been trained yet.')
            results = None
            true_predictions = None
        else:
            # We tokenize our texts to use them in BERT
            test_encodings = self._tokenize_texts(test_texts)

            if test_labels:
                test_labels_encoded = self._encode_tags(test_labels, test_encodings)
                test_dataset = TorchDataset(test_encodings, test_labels_encoded)

                predictions, labels, _ = self._trainer.predict(test_dataset)
                align_predictions, align_labels = self._realign_overflowing_tokens(test_encodings, labels, predictions)
                predictions = [np.argmax(doc_predictions, axis=1) for doc_predictions in align_predictions]

                # We need to remove the unwanted tokens, written as -100 label. But when we reconstruct, if the first label of
                # a new document is not the first token of a word it should have a label of -100 but the label is 0.
                # This line works if the number of classes is inferior at 50.
                true_predictions = [
                    [self._id2tag[p] for (p, l) in zip(prediction, label) if l!=IGNORED_TOKEN]
                    for prediction, label in zip(predictions, align_labels)
                ]

                results = self._metric.compute(predictions=true_predictions, references=test_labels)

            else:
                # We need to create fake labels to realign the predicted labels and the original text
                test_fake_labels = [['O' for word in text] for text in test_texts]
                test_fake_labels = self._encode_tags(test_fake_labels, test_encodings)

                test_dataset = TorchDataset(test_encodings)
                predictions, labels, _ = self._trainer.predict(test_dataset)
                align_predictions, align_labels = self._realign_overflowing_tokens(test_encodings, test_fake_labels, predictions)
                predictions = [np.argmax(doc_predictions, axis=1) for doc_predictions in align_predictions]

                # We need to remove the unwanted tokens, written as -100 label.
                true_predictions = [
                    [self._id2tag[p] for (p, l) in zip(prediction, label) if l!=IGNORED_TOKEN]
                    for prediction, label in zip(predictions, align_labels)
                ]

                results = None

            return results, true_predictions

    def _tokenize_texts(self, texts):
        """Tokenize the texts.

        Args:
            texts (str): pre-tokenized texts with each words an element of a list

        Returns:
            Encoding: the tokenized texts
        """
        texts = [[word.replace("\x97", "-") for word in text] for text in texts]
        tokenized_texts = self._tokenizer(texts, is_split_into_words=True, padding="max_length", truncation=True,
                                          return_overflowing_tokens=True, stride=self._stride, max_length=self._max_length)
        return tokenized_texts    

    def _encode_tags(self, tags, encodings):
        """Transform tags in ids and taking in account the tokenizer.

        Args:
            tags (list of list of str): list of list of tags
            encodings (BatchEncoding): HuggingFace tokenized text 

        Returns:
            list of list of int: transforming tags in int.
        """
        encoded_tags = []
        # We look at the number of encoded documents as they can be more than real documents
        for i, current_doc in enumerate(encodings['overflow_to_sample_mapping']):
            word_ids = encodings.word_ids(batch_index=i)

            # We take the good document in function of the overflow mapping
            tag = tags[current_doc]

            previous_word_idx = None
            tag_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the tag to -100 so they are automatically
                # ignored in the loss function. Base value.
                if word_idx is None:
                    tag_ids.append(IGNORED_TOKEN)
                # We set the tag for the first token of each word.
                elif word_idx != previous_word_idx:
                    tag_ids.append(self._tag2id[tag[word_idx]])
                # For the other tokens in a word, we set the tag to -100
                else:
                    tag_ids.append(IGNORED_TOKEN)
                previous_word_idx = word_idx

            encoded_tags.append(tag_ids)

        return encoded_tags

    def _compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)
        # Remove ignored index (special tokens)
        true_predictions = [
            [self._id2tag[p] for (p, l) in zip(prediction, label) if l!=IGNORED_TOKEN]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self._id2tag[l] for (p, l) in zip(prediction, label) if l!=IGNORED_TOKEN]
            for prediction, label in zip(predictions, labels)
        ]

        results = self._metric.compute(predictions=true_predictions, references=true_labels)

        # Computing micro F1 score with the outside labels.
        mlb = MultiLabelBinarizer()
        full_f1_score = f1_score(mlb.fit_transform(true_labels), mlb.transform(true_predictions), average="micro")

        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "f1_with_outside": full_f1_score
        }

    def _realign_overflowing_tokens(self, encodings, labels, predictions):
        """Realign overflowing labels and predictions into one list per document.

        Args:
            encodings (Encodings): encoding of the data using a Transformer tokenizer.
            labels (list): list of list of int, label of each token.
            predictions (list): the predictions for token for each class.

        Returns:
            pred, labels: realign predictions and labels
        """
        overflowing_docs = encodings['overflow_to_sample_mapping']
        align_labels = []
        align_predictions = []

        current_doc_id = -1
        current_labels = []
        current_predictions = []

        for i, doc_id in enumerate(overflowing_docs):
            if doc_id != current_doc_id:
                if current_doc_id != -1:
                    align_labels.append(current_labels)
                    align_predictions.append(current_predictions)
                current_labels = labels[i]
                current_predictions = predictions[i]
                current_doc_id = doc_id
            
            else:
                # We remove the last token as it's an end token and the first token of the overflow as it's a begin token
                current_labels = self._merge_overflow_list(current_labels[:-1], labels[i][1:], self._stride, True)
                current_predictions = self._merge_overflow_list(current_predictions[:-1], predictions[i][1:], self._stride, False)

        align_labels.append(current_labels)
        align_predictions.append(current_predictions)    
        
        return align_predictions, align_labels
                
    def _merge_overflow_list(self, normal_list, overflow_list, stride, is_labels):
        """Merge a list with it's overflow.

        Args:
            normal_list (list): The beginning of the list
            overflow_list (list): The overflow
            stride (int): the number of tokens in common between the two lists
            is_labels(bool): if we are replacing labels or not
        Returns:
            [type]: [description]
        """
        before_list = normal_list[:-stride]
        common_normal_list = normal_list[-stride:]
        common_overflow_list = overflow_list[:stride]
        after_list = overflow_list[stride:]

        if is_labels:
            common_list = np.min([common_normal_list, common_overflow_list], axis=0)
        else:
            common_list = np.mean([common_normal_list, common_overflow_list], axis=0)

        final_list = np.concatenate((before_list, common_list, after_list))

        return final_list
