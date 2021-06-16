"""The goal of this file is to train a BERT model to resolve the first task."""

import click
import datetime
import os
from pathlib import Path
import pickle

import mlflow
from transformers import TrainingArguments
from sklearn.model_selection import train_test_split

from event_extraction.model.bert_sequence_classif import BertSequenceClassif
from event_extraction.data_loader.protest_news2019 import ProtestNews2019

@click.command()
@click.option("--epochs", default=20, help="Number of epochs.")
@click.option("--learning_rate", default=5e-04, help="Starting learning rate.")
@click.option("--train_batch_size", default=6, help="Batch size for training per device.")
@click.option("--eval_batch_size", default=8, help="Batch size for evaluating per device.")
@click.option("--model_name", default="distilbert-base-uncased", help="Name of the model in the HF modelhub or local path.")
def main(epochs, learning_rate, train_batch_size, eval_batch_size, model_name):
    """Run the model."""
    # Set active experiement
    mlflow.set_experiment("ProtestNews2019_task1")

    # Choosing the folder to gather all the results
    results_dir = Path('results/task_1')
    results_dir = results_dir / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Loading the data
    data_loader = ProtestNews2019('data/task_2019')
    data_dict = data_loader.load_task1()

    train_texts, val_texts, train_labels, val_labels = \
        train_test_split(data_dict['train_texts'], data_dict['train_labels'], test_size=.2, random_state=13)

    # Specify the arguments
    training_arguments = TrainingArguments(output_dir=results_dir, num_train_epochs=epochs, learning_rate=learning_rate, 
                                           per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,
                                           metric_for_best_model="eval_loss", load_best_model_at_end=True, 
                                           evaluation_strategy="epoch", logging_strategy="epoch", save_total_limit=0
                                           )

    # Training the model
    trainer = BertSequenceClassif(pretrained_model=model_name, training_arguments=training_arguments)
    trainer.train(train_texts, train_labels,
                val_texts, val_labels)

    # Evaluate on the test datasets
    results, predictions = trainer.predict(data_dict['eval_texts'], data_dict['eval_labels'])
    mlflow.log_param("dataset", "task1_2019_train_eval")

    # Create folder for the results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Saving the best model
    trainer.save_model(results_dir / "model")

    mlflow.log_artifacts(results_dir / "model", artifact_path="model")
    
    # Saving results to pickle
    eval_file_path = results_dir / "eval.results"
    with open(eval_file_path, 'wb') as eval_file:
        pickle.dump(results, eval_file)


if __name__ == '__main__':
    main()
