"""Test BERT_token_classif.py file."""
import logging

import mlflow
from transformers import TrainingArguments

from event_extraction.model.bert_sequence_classif import BertSequenceClassif
from event_extraction.data_loader.loader import Loader

logger = logging.getLogger()

def test_training_testing():
    """Test if the training and testing works."""
    loader = Loader()
    text, label = loader.read_json("tests/test_data/train_filled.json")

    training_arguments = TrainingArguments(output_dir="tests/results_test", num_train_epochs=1, save_strategy="no",
                                           per_device_train_batch_size=1, per_device_eval_batch_size=1)

    classifier = BertSequenceClassif(pretrained_model="julien-c/bert-xsmall-dummy",
                                     training_arguments=training_arguments)

    logger.info("Classifier created, starting training.")

    classifier.train(text, label, text, label)

    logger.info('Training done, starting prediction.')

    classifier.predict(text, label)

def test_long_text():
    """Test if long text works."""
    texts = ["test "*600, "je suis un test"]
    labels = [0, 1]

    training_arguments = TrainingArguments(output_dir="tests/results_test", num_train_epochs=1, save_strategy="no",
                                           per_device_train_batch_size=1, per_device_eval_batch_size=1)

    classifier = BertSequenceClassif(pretrained_model="julien-c/bert-xsmall-dummy",
                                     training_arguments=training_arguments)
    classifier.train(texts, labels, texts, labels)

    _, results_labels = classifier.predict(texts, labels)
    _, results_no_labels = classifier.predict(texts)

    assert len(texts) == len(results_labels)
    assert len(texts) == len(results_no_labels)