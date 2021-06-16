"""Test BERT_token_classif.py file."""
import logging

import mlflow
from transformers import TrainingArguments

from event_extraction.data_loader.protest_news2019 import ProtestNews2019
from event_extraction.model.bert_token_classif import BertTokenClassif

logger = logging.getLogger()
mlflow.set_experiment("Test")

def test_encode_decode_equivalence():
    """Test if we can decode the encoding."""
    tag2id = {'O': 0, 'I': 1, 'B': 2}
    id2tag = {0: 'O', 1:'I', 2:'B'}

    texts = [['I', 'am', 'going', 'to', 'New', 'York'], ['Marseille', 'est', 'magnifique', 'motetrangequinexistepas', 'Louvain']]
    labels = [['O', 'O', 'O', 'O', 'B', 'I'], ['B', 'O', 'O', 'O', 'B']]

    training_arguments = TrainingArguments(output_dir="tests/results_test", num_train_epochs=1, save_strategy="no")

    test_classif = BertTokenClassif(pretrained_model="julien-c/bert-xsmall-dummy",
                                    training_arguments=training_arguments)

    # We don't want to learn a model so we force the dictionaries
    test_classif._tag2id = tag2id
    test_classif._id2tag = id2tag

    texts_encodings = test_classif._tokenize_texts(texts)

    encoded_labels = test_classif._encode_tags(labels, texts_encodings)

    decoded_labels = [
            [id2tag[l] for l in label if l != -100]
            for label in encoded_labels
        ]

    assert labels == decoded_labels

def test_training_testing():
    """Test if the training and testing works."""
    data_loader = ProtestNews2019('tests/test_data')
    data_tests = data_loader.load_task3()

    training_arguments = TrainingArguments(output_dir="tests/results_test", num_train_epochs=2,
                                           evaluation_strategy="epoch", save_strategy="no")

    classifier = BertTokenClassif(pretrained_model="julien-c/bert-xsmall-dummy",
                                  training_arguments=training_arguments)

    logger.info("Classifier created, starting training.")

    classifier.train(data_tests['train_texts'], data_tests['train_labels'],
                     data_tests['eval_texts'], data_tests['eval_labels'])

    logger.info('Training done, starting prediction.')

    classifier.predict(data_tests['test_texts'], data_tests['test_labels'])

def test_training_long_doc():
    """Test if it works on doc longer than maximum."""
    texts = [["test"]*600, ["Kilimandjaro"]*300, ["je", "suis", "un", "test"], ["un"]+["unun"]*600]
    labels = [["O"]*600, ["O"]*300, ["B", "I", "I", "I"], ["O"]*601]

    training_arguments = TrainingArguments(output_dir="tests/results_test", num_train_epochs=1, save_strategy="no")

    classifier = BertTokenClassif(pretrained_model="nlpaueb/legal-bert-small-uncased",
                                  training_arguments=training_arguments)
    classifier.train(texts, labels, texts, labels)

    _, results_labels = classifier.predict(texts, labels)
    _, results_no_labels = classifier.predict(texts)

    assert all([len(text) == len(result) for text, result in zip(texts, results_labels)])
    assert all([len(text) == len(result) for text, result in zip(texts, results_no_labels)])

def test_loading_local_model():
    """Test if the training and testing works with a local model."""
    data_loader = ProtestNews2019('tests/test_data')
    data_tests = data_loader.load_task3()

    training_arguments = TrainingArguments(output_dir="tests/results_test", num_train_epochs=2, per_device_eval_batch_size=5,
                                           evaluation_strategy="epoch", logging_strategy="epoch", do_eval=True,
                                           save_strategy="no")

    classifier = BertTokenClassif(pretrained_model="./tests/test_data/dummy_model/",
                                  training_arguments=training_arguments, initialization_prediction_layer_seed=513)

    logger.info("Classifier created, starting training.")

    classifier.train(data_tests['train_texts'], data_tests['train_labels'],
                     data_tests['eval_texts'], data_tests['eval_labels'])

    logger.info('Training done, starting prediction.')

    classifier.predict(data_tests['test_texts'], data_tests['test_labels'])

test_loading_local_model()