"""The goal of this file is to train a BERT model to resolve the third task of 2019 ProtestNews."""

import click
import datetime
import os
from pathlib import Path
import pickle

import mlflow
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

from event_extraction.model.bert_token_classif import BertTokenClassif
from event_extraction.data_loader.protest_news2019 import ProtestNews2019
from event_extraction.statistic.eval_token_classif import EvalTokenClassif

@click.command()
@click.option("--epochs", default=20, help="Number of epochs.")
@click.option("--learning_rate", default=5e-04, help="Starting learning rate.")
@click.option("--train_batch_size", default=6, help="Batch size for training per device.")
@click.option("--eval_batch_size", default=8, help="Batch size for evaluating per device.")
@click.option("--model_name", default="distilbert-base-cased", help="Name of the model in the HF modelhub or local path.")
def main(epochs, learning_rate, train_batch_size, eval_batch_size, model_name):
    """Run the model."""
    # Setting the experiment
    mlflow.set_experiment("ProtestNews2019_task3")

    # Choosing the folder to gather all the results
    results_dir = Path('results/2019_task_3')
    results_dir = results_dir / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Loading the data
    data_loader = ProtestNews2019('data/task_2019')
    data_dict = data_loader.load_task3()

    train_texts, val_texts, train_labels, val_labels = \
        train_test_split(data_dict['train_texts'], data_dict['train_labels'], test_size=.2, random_state=13)

    # Specify the arguments
    training_arguments = TrainingArguments(output_dir=results_dir, num_train_epochs=epochs, learning_rate=learning_rate, 
                                           per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,
                                           load_best_model_at_end=True, metric_for_best_model="eval_f1", greater_is_better=True,
                                           evaluation_strategy="epoch", logging_strategy="epoch", save_total_limit=0,
                                           )

    # Training the model
    trainer = BertTokenClassif(pretrained_model=model_name, training_arguments=training_arguments)
    trainer.train(train_texts, train_labels,
                  val_texts, val_labels)

    # Evaluate on the test datasets
    results_dev, labels_predict = trainer.predict(data_dict['eval_texts'], data_dict['eval_labels'])
    mlflow.log_param("dataset", "task3_2019_train_eval")

    # Create folder for the results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Saving the best model
    trainer.save_model(results_dir / "model")

    mlflow.log_artifacts(results_dir / "model", artifact_path="model")

    # Printing results to files
    dev_file = results_dir / "dev.results"
    with dev_file.open('wb') as results_pickle:
        pickle.dump(results_dev, results_pickle)

    txt_results = results_dir / "predicted_results.txt"
    with txt_results.open('w') as results_txt:
        # From two different lists to each word and it label together.
        merge_text_labels = [[(word, label) for word, label in zip(text, label_doc)] for text, label_doc  in zip (data_dict['eval_texts'], labels_predict)]
        for docs in merge_text_labels:
            for word, label in docs:
                results_txt.write('{}\t{}\n'.format(word, label))
            results_txt.write('\n')

    mlflow.log_artifact(txt_results)

    evaluator = EvalTokenClassif(results_dir)
    score_results = evaluator.evaluate('data/task3public9may/dev.txt', txt_results)
    mlflow.log_metric("test_precision", score_results[0])
    mlflow.log_metric("test_recall", score_results[1])
    mlflow.log_metric("test_f1", score_results[2])
    score_results_txt = results_dir / "score_results.txt"
    with score_results_txt.open('w') as score_results_file:
        score_results_file.write(str(score_results))

    mlflow.log_artifact(score_results_txt)

if __name__ == '__main__':
    main()
