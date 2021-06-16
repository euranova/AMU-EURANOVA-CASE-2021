"""The goal of this file is to train a BERT model to resolve the 1st subtask of task 1 of 2021 ProtestNews (event-extraction)."""
import click
import datetime
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from transformers import TrainingArguments

from event_extraction.data_loader.loader import Loader
from event_extraction.model.bert_sequence_classif import BertSequenceClassif

def shuffle_texts_labels(texts, labels, random_generator):
    """Shuffle texts and labels into a new order while keepoing the texts-labels correspondance.

    Args:
        texts (list): texts to sort.
        labels (list): labels to sort.
        random_generator (np.random.RandomState): the random generator.

    Returns:
        texts, labels: randomly sorted texts, labels.
    """
    data = list(zip(texts, labels))
    random_generator.shuffle(data)
    texts, labels = list(zip(*data))
    return list(texts), list(labels)

@click.command()
@click.option("--data_dir", default="data", help="Directory with the data.")
@click.option("--epochs", default=1, help="Number of epochs.")
@click.option("--learning_rate", default=5e-04, help="Starting learning rate.")
@click.option("--train_batch_size", default=6, help="Batch size for training per device.")
@click.option("--eval_batch_size", default=8, help="Batch size for evaluating per device.")
@click.option("--model_name", default="bert-base-multilingual-cased", help="Name of the model in the HF modelhub or local path.")
def main(data_dir, epochs, learning_rate, train_batch_size, eval_batch_size, model_name):
    results_dir = Path("results/2021_subtask_1")
    results_dir = results_dir / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    data_dir = Path(data_dir)

    loader = Loader()

    train_en_texts, train_en_labels = loader.read_json(data_dir / "subtask1-document/en-train.json")
    train_es_texts, train_es_labels = loader.read_json(data_dir / "subtask1-document/es-train.json")
    train_pt_texts, train_pt_labels = loader.read_json(data_dir / "subtask1-document/pr-train.json")

    # We want all the splits to contain the same amount of texts in each language.
    train_texts_en, eval_texts_en, train_labels_en, eval_labels_en = \
        train_test_split(train_en_texts, train_en_labels, test_size=.2, random_state=13)

    train_texts_es, eval_texts_es, train_labels_es, eval_labels_es = \
        train_test_split(train_es_texts, train_es_labels, test_size=.2, random_state=13)

    train_texts_pt, eval_texts_pt, train_labels_pt, eval_labels_pt = \
        train_test_split(train_pt_texts, train_pt_labels, test_size=.2, random_state=13)

    # We now create our splits with all the languages.
    train_texts = train_texts_en + train_texts_es + train_texts_pt
    eval_texts = eval_texts_en + eval_texts_es + eval_texts_pt
    train_labels = train_labels_en + train_labels_es + train_labels_pt
    eval_labels = eval_labels_en + eval_labels_es + eval_labels_pt

    # We shuffle the list and labels to shuffle the languages between them.
    rng = np.random.RandomState(seed=13)
    train_texts, train_labels = shuffle_texts_labels(train_texts, train_labels, rng)
    eval_texts, eval_labels = shuffle_texts_labels(eval_texts, eval_labels, rng)

    # We can now import the test data.
    with open(data_dir / "english/subtask1-Document/test.json", "r") as data_file:
        data = data_file.read()
    test_data_en = [json.loads(elem) for elem in data.split('\n')]
    with open(data_dir / "hindi/subtask1-Document/test.json", "r") as data_file:
        data = data_file.read()
    test_data_hi = [json.loads(elem) for elem in data.split('\n')]
    with open(data_dir / "portuguese/subtask1-Document/test.json", "r") as data_file:
        data = data_file.read()
    test_data_pt = [json.loads(elem) for elem in data.split('\n')]
    with open(data_dir / "spanish/subtask1-Document/test.json", "r") as data_file:
        data = data_file.read()
    test_data_es = [json.loads(elem) for elem in data.split('\n')]

    # Specify the arguments
    training_arguments = TrainingArguments(output_dir=results_dir, num_train_epochs=epochs, learning_rate=learning_rate, 
                                           per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,
                                           load_best_model_at_end=True, greater_is_better=True, eval_steps=100,
                                           evaluation_strategy="steps", logging_strategy="steps", save_total_limit=1,
                                          )
    
    # Training the model
    trainer = BertSequenceClassif(pretrained_model=model_name, training_arguments=training_arguments)
    trainer.train(train_texts, train_labels, eval_texts, eval_labels)

    test_texts_en = [test_doc['text'] for test_doc in test_data_en]
    test_texts_hi = [test_doc['text'] for test_doc in test_data_hi]
    test_texts_pt = [test_doc['text'] for test_doc in test_data_pt]
    test_texts_es = [test_doc['text'] for test_doc in test_data_es]

    _, results_tests_en = trainer.predict(test_texts_en)
    _, results_tests_hi = trainer.predict(test_texts_hi)
    _, results_tests_pt = trainer.predict(test_texts_pt)
    _, results_tests_es = trainer.predict(test_texts_es)

    json_results_en = [{"id": test_doc["id"], "prediction": int(pred)} for test_doc, pred in zip(test_data_en, results_tests_en)]
    json_results_hi = [{"id": test_doc["id"], "prediction": int(pred)} for test_doc, pred in zip(test_data_hi, results_tests_hi)]
    json_results_pt = [{"id": test_doc["id"], "prediction": int(pred)} for test_doc, pred in zip(test_data_pt, results_tests_pt)]
    json_results_es = [{"id": test_doc["id"], "prediction": int(pred)} for test_doc, pred in zip(test_data_es, results_tests_es)]

    # Create folder for the results
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "english/").mkdir(parents=True, exist_ok=True)
    (results_dir / "hindi/").mkdir(parents=True, exist_ok=True)
    (results_dir / "portuguese/").mkdir(parents=True, exist_ok=True)
    (results_dir / "spanish/").mkdir(parents=True, exist_ok=True)

    json_file_en = results_dir / "english/submission.json"
    json_file_hi = results_dir / "hindi/submission.json"
    json_file_pt = results_dir / "portuguese/submission.json"
    json_file_es = results_dir / "spanish/submission.json"

    with open(json_file_en, "w") as json_path_en:
        json_path_en.write(str(json_file_en))

    with open(json_file_hi, "w") as json_path_hi:
        json_path_hi.write(str(json_file_hi))

    with open(json_file_pt, "w") as json_path_pt:
        json_path_pt.write(str(json_file_pt))

    with open(json_file_es, "w") as json_path_es:
        json_path_es.write(str(json_file_es))

if __name__ == "__main__":
    main()