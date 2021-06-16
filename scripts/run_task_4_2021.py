"""The goal of this file is to train a BERT model to resolve the 4th task of 2021 ProtestNews (event-extraction)."""
import datetime
from pathlib import Path

import click
from datasets import load_metric
import mlflow
from ray import tune
from ray.tune.integration.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import MedianStoppingRule
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from transformers import TrainingArguments

from event_extraction.model.bert_token_classif import BertTokenClassif
from event_extraction.data_loader.task4_2021 import Task4_2021
from event_extraction.data_loader.loader import Loader
from event_extraction.statistic.eval_token_classif import EvalTokenClassif

def print_texts_labels_to_file(texts, labels, path_file):
    """Print texts labels to file.

    Args:
        texts (list): texts of the documents.
        labels (list of list): labels of all the tokens of all the documents.
        path_file (Path): path where the file will be written.
    """
    with path_file.open('w') as results_txt:
        # From two different lists to each word and it label together.
        merge_text_labels = [[(word, label) for word, label in zip(text, label_doc)] for text, label_doc  in zip (texts, labels)]
        for docs in merge_text_labels:
            for word, label in docs:
                results_txt.write('{}\t{}\n'.format(word, label))
            results_txt.write('\n')

def get_eval_f1(metrics):
        """Return the eval_F1, useful for hyperparameter search.

        Args:
            metrics (Dict): dictionnary with all the metrics.

        Returns:
            float: the F1 on the evaluation dataset.
        """
        return metrics["eval_f1"]

@click.command()
@click.option("--epochs", default=1, help="Number of epochs.")
@click.option("--learning_rate", default=5e-04, help="Starting learning rate.")
@click.option("--train_batch_size", default=6, help="Batch size for training per device.")
@click.option("--eval_batch_size", default=8, help="Batch size for evaluating per device.")
@click.option("--model_name", default="bert-base-multilingual-cased", help="Name of the model in the HF modelhub or local path.")
@click.option("--loss", default="macro", help="either 'base', 'micro' or 'macro'")
@click.option("--hyperparameter_search", default=0)
@click.option("--n_trials", default=20, help="Number of trials for hyperparameter search.")
@click.option("--data_order_seed", default=13, help="Seed to change order of the data.")
@click.option("--initialization_prediction_layer_seed", default=13, help="Seed to change initialization of predictio layer.")
@click.option("--dataset", default="full", help="full, full_eval or train_only")
def main(epochs, learning_rate, train_batch_size, eval_batch_size, model_name, loss, hyperparameter_search, n_trials,
         data_order_seed=13, initialization_prediction_layer_seed=13, dataset="full"):
    """Run the model."""
    # Choosing the folder to gather all the results
    results_dir = Path("results/2021_task_4")
    results_dir = results_dir / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Loading the data
    path_data = Path("data/task_2021/subtask4-token")
    if dataset == "train_only":
        data_loader = Task4_2021(path_data=path_data, data_order_seed=data_order_seed)
        train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels = \
            data_loader.load_dataset()
        train_texts_en, train_labels_en, eval_texts_en, eval_labels_en, test_texts_en, test_labels_en = \
        data_loader.load_dataset_en()
        train_texts_es, train_labels_es, eval_texts_es, eval_labels_es, test_texts_es, test_labels_es = \
            data_loader.load_dataset_es()
        train_texts_pt, train_labels_pt, eval_texts_pt, eval_labels_pt, test_texts_pt, test_labels_pt = \
            data_loader.load_dataset_pt()
    elif dataset == "full_eval":
        data_loader = Task4_2021(path_data=path_data, data_order_seed=data_order_seed)
        train_texts, train_labels, eval_texts, eval_labels, _, _ = \
            data_loader.load_dataset(test_size=0)
        data_loader_test = Loader()
        test_texts_en, test_labels_en = data_loader_test.read_token_annotated_file("data/task_2021/english/subtask4-Token/test.txt")
        test_texts_es, test_labels_es = data_loader_test.read_token_annotated_file("data/task_2021/spanish/subtask4-Token/test.txt")
        test_texts_pt, test_labels_pt = data_loader_test.read_token_annotated_file("data/task_2021/portuguese/subtask4-Token/test.txt")
        test_texts = test_texts_en + test_texts_es + test_texts_pt
        test_labels = None
    
    elif dataset == "full":
        data_loader = Task4_2021(path_data=path_data, data_order_seed=data_order_seed)
        train_texts, train_labels, eval_texts, eval_labels, _, _ = \
            data_loader.load_dataset(eval_size=0, test_size=0)
        data_loader_test = Loader()
        test_texts_en, test_labels_en = data_loader_test.read_token_annotated_file("data/task_2021/english/subtask4-Token/test.txt")
        test_texts_es, test_labels_es = data_loader_test.read_token_annotated_file("data/task_2021/spanish/subtask4-Token/test.txt")
        test_texts_pt, test_labels_pt = data_loader_test.read_token_annotated_file("data/task_2021/portuguese/subtask4-Token/test.txt")

        test_texts = test_texts_en + test_texts_es + test_texts_pt
        test_labels = None

    else:
        raise ValueError("Argument dataset is neither full, full_eval or train_only")

    if eval_texts:
        do_eval=True
        evaluation_strategy="epoch"

    else:
        do_eval=False
        evaluation_strategy="no"

    # Basic training
    if not hyperparameter_search:
        # Parameters 
        training_arguments = TrainingArguments(output_dir=results_dir, num_train_epochs=epochs, learning_rate=learning_rate, 
                                           per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,
                                           load_best_model_at_end=True, metric_for_best_model="eval_f1", greater_is_better=True,
                                           evaluation_strategy=evaluation_strategy, logging_strategy="epoch", save_total_limit=3,
                                           weight_decay=0.36, max_grad_norm=0.17, adam_epsilon=3e-08, adam_beta2=0.99, adam_beta1=0.74,
                                           adafactor=True, do_eval=do_eval
                                           )
        trainer = BertTokenClassif(pretrained_model=model_name, training_arguments=training_arguments, loss=loss,
                                   initialization_prediction_layer_seed=initialization_prediction_layer_seed)
        trainer.train(train_texts, train_labels, eval_texts, eval_labels)
    # Hyperparameter_search
    else:
        training_arguments = TrainingArguments(output_dir=results_dir, num_train_epochs=epochs, learning_rate=learning_rate, 
                                           per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,
                                           load_best_model_at_end=True, metric_for_best_model="eval_f1", greater_is_better=True,
                                           evaluation_strategy=evaluation_strategy, logging_strategy="epoch", save_total_limit=3, do_eval=do_eval,
                                           report_to="none", disable_tqdm=True, # We need to disable tqdm or logs are unreadable
                                           )
        trainer = BertTokenClassif(pretrained_model=model_name, training_arguments=training_arguments, loss=loss,
                                   initialization_prediction_layer_seed=initialization_prediction_layer_seed)
        # Defining the research space for hyperparameters
        hp_space = {
            "num_train_epochs":tune.choice([20, 25, 30, 40]),
            "weight_decay": tune.uniform(0.0001, 1),
            "learning_rate": tune.choice([1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 2e-7, 1e-7, 3e-7, 2e-8]),
            "adafactor": tune.choice(['True','False']),
            "adam_beta1": tune.uniform(0.0, 1.0),
            "adam_beta2": tune.uniform(0, 1.0),
            "adam_epsilon": tune.choice([1e-8, 2e-8, 3e-8, 1e-9, 2e-9, 3e-10]),
            "max_grad_norm": tune.uniform(0, 1.0),
        }
        # Adding good hyperparameters to simplify the start, this results have been obtained during previous hyperparameter-search
        current_best_params = [{
            "num_train_epochs":40,
            "weight_decay":0.76,
            "learning_rate":4e-05,
            "adafactor":'False',
            "adam_beta1":0.17,
            "adam_beta2":0.88,
            "adam_epsilon":3e-08,
            "max_grad_norm":0.17
        },
        {
            "num_train_epochs":40,
            "weight_decay":0.36,
            "learning_rate":5e-05,
            "adafactor":'True',
            "adam_beta1":0.74,
            "adam_beta2":0.99,
            "adam_epsilon":3e-08,
            "max_grad_norm":0.77
        },
        ]
        resources_per_trial={
            "cpu":12,
            "gpu":1
        }

        # Defining elements of ray.
        if do_eval:
            metric = "eval_f1"
        else:
            metric = "train_f1"
        search_alg = HyperOptSearch(metric=metric, mode="max", n_initial_points=5, points_to_evaluate=current_best_params)
        search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1, batch=True)

        scheduler = MedianStoppingRule(time_attr='training_iteration', # it's the epoch in fact
                                  metric=metric,
                                  mode='max',
                                  grace_period=3,
                                  min_samples_required=3,
                                  )

        best_run = trainer.hyperparameter_search(train_texts, train_labels, eval_texts, eval_labels, hp_space=hp_space,
                                                 n_trials=n_trials,direction="maximize", backend="ray", search_alg=search_alg,
                                                 resources_per_trial=resources_per_trial, compute_objective=get_eval_f1,
                                                 callbacks=[MLflowLoggerCallback(experiment_name="hs_2021_task4", save_artifact=True)])

        mlflow.log_param("Best run place", best_run.run_id)
        mlflow.log_param("Best hyperparameters", best_run.hyperparameters)
        mlflow.log_metric("Best objective score (eval F1 score)", best_run.objective)

    # Evaluate on the test datasets
    _, predicted_labels = trainer.predict(test_texts, test_labels)
    mlflow.log_param("dataset", dataset)

    _, predicted_labels_en = trainer.predict(test_texts_en, test_labels_en)
    _, predicted_labels_es = trainer.predict(test_texts_es, test_labels_es)
    _, predicted_labels_pt = trainer.predict(test_texts_pt, test_labels_pt)

    if test_labels_en and test_labels_es and test_labels_pt:
        metric = load_metric("seqeval", zero_division=0)
        results_en = metric.compute(predictions=predicted_labels_en, references=test_labels_en)
        results_es = metric.compute(predictions=predicted_labels_es, references=test_labels_es)
        results_pt = metric.compute(predictions=predicted_labels_pt, references=test_labels_pt)
        results_full = metric.compute(predictions=predicted_labels, references=test_labels)

        mlflow.log_metrics({
            "test_f1_en": results_en["overall_f1"],
            "test_f1_es": results_es["overall_f1"],
            "test_f1_pt": results_pt["overall_f1"],
            "test_f1_full": results_full["overall_f1"]
        })

    # Create folder for the results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Saving the best model
    trainer.save_model(results_dir / "model")
    mlflow.log_artifacts(results_dir / "model", artifact_path="model")

    # Printing the predicted results to file
    txt_results = results_dir / "predicted_results_en.txt"
    print_texts_labels_to_file(test_texts_en, predicted_labels_en, txt_results)

    txt_results = results_dir / "predicted_results_es.txt"
    print_texts_labels_to_file(test_texts_es, predicted_labels_es, txt_results)

    txt_results = results_dir / "predicted_results_pt.txt"
    print_texts_labels_to_file(test_texts_pt, predicted_labels_pt, txt_results)

    txt_results = results_dir / "predicted_results.txt"
    print_texts_labels_to_file(test_texts, predicted_labels, txt_results)

    if dataset == "train_only":
        # Printing the gold results to file for evaluator
        txt_gold = results_dir / "gold_results.txt"
        print_texts_labels_to_file(test_texts, test_labels, txt_gold)
        evaluator = EvalTokenClassif(results_dir)
        score_results = evaluator.evaluate(txt_gold, txt_results)
        score_results_txt = results_dir / "score_results.txt"
        mlflow.log_artifact(score_results_txt)
        mlflow.log_artifact(results_dir / "score_plot.png")

    mlflow.log_artifact(txt_results)

if __name__ == '__main__':
    main()
