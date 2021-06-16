"""Script running the behaviroual pretraining for task4."""
import datetime
from pathlib import Path

import click
from datasets import load_metric
import mlflow
from transformers import TrainingArguments

from event_extraction.data_loader.multiling_ner import MutlilingNER
from event_extraction.data_loader.multling_ner_wiki import MutlilingNERWiki
from event_extraction.model.bert_token_classif import BertTokenClassif

@click.command()
@click.option("--epochs", default=1, help="Number of epochs.")
@click.option("--learning_rate", default=5e-04, help="Starting learning rate.")
@click.option("--train_batch_size", default=6, help="Batch size for training per device.")
@click.option("--eval_batch_size", default=8, help="Batch size for evaluating per device.")
@click.option("--model_name", default="bert-base-multilingual-cased", help="Name of the model in the HF modelhub or local path.")
@click.option("--loss", default="macro", help="either 'base', 'micro' or 'macro'")
@click.option("--hyperparameter_search", default=0)
@click.option("--n_trials", default=20, help="Number of trials for hyperparameter search.")
@click.option("--dataset", default="multi_ner", help="Name of the dataset, multi_ner or multi_ner_wiki")
def main(epochs, learning_rate, train_batch_size, eval_batch_size, model_name, loss, hyperparameter_search, n_trials, dataset):
    """Enable behavioural pretraining."""
    # Choosing the folder to gather all the results
    results_dir = Path("results/2021_task_4_behavioural")
    results_dir = results_dir / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Part 1: training on the big datasets
    if dataset == "multi_ner":
        data_loader = MutlilingNER()
        
    elif dataset == "multi_ner_wiki":
        data_loader = MutlilingNERWiki()

    else:
        raise ValueError("dataset is not multi_ner or multi_ner_wiki")

    train_texts, train_labels, eval_texts, eval_labels, test_texts, test_labels = \
            data_loader.load_dataset()


    training_arguments = TrainingArguments(output_dir=results_dir, num_train_epochs=epochs, learning_rate=learning_rate,
                                           per_device_train_batch_size=train_batch_size, per_device_eval_batch_size=eval_batch_size,
                                           load_best_model_at_end=True, metric_for_best_model="eval_f1", greater_is_better=True,
                                           evaluation_strategy="epoch", logging_strategy="epoch", save_total_limit=1,
                                           adafactor=True, adam_epsilon=3e-08, adam_beta1=0.77, adam_beta2=0.82,
                                           max_grad_norm=0.74, weight_decay=0.11
                                          )
    trainer = BertTokenClassif(pretrained_model=model_name, training_arguments=training_arguments, loss=loss)
    trainer.train(train_texts, train_labels, eval_texts, eval_labels)

    # Evaluate on the test datasets
    _, predicted_labels = trainer.predict(test_texts, test_labels)
    mlflow.log_param("dataset", dataset)
    metric = load_metric("seqeval", zero_division=0)
    results = metric.compute(predictions=predicted_labels, references=test_labels)

    mlflow.log_metric("test_f1", results["overall_f1"])

    train_texts_en, train_labels_en, eval_texts_en, eval_labels_en, test_texts_en, test_labels_en = \
        data_loader.load_dataset_en()

    train_texts_es, train_labels_es, eval_texts_es, eval_labels_es, test_texts_es, test_labels_es = \
        data_loader.load_dataset_es()

    train_texts_pt, train_labels_pt, eval_texts_pt, eval_labels_pt, test_texts_pt, test_labels_pt = \
        data_loader.load_dataset_pt()

    _, predicted_labels_en = trainer.predict(test_texts_en, test_labels_en)
    _, predicted_labels_es = trainer.predict(test_texts_es, test_labels_es)
    _, predicted_labels_pt = trainer.predict(test_texts_pt, test_labels_pt)

    results_en = metric.compute(predictions=predicted_labels_en, references=test_labels_en)
    results_es = metric.compute(predictions=predicted_labels_es, references=test_labels_es)
    results_pt = metric.compute(predictions=predicted_labels_pt, references=test_labels_pt)

    mlflow.log_metrics({
        "test_f1_en": results_en["overall_f1"],
        "test_f1_es": results_es["overall_f1"],
        "test_f1_pt": results_pt["overall_f1"]
    })

    # Create folder for the results
    results_dir.mkdir(parents=True, exist_ok=True)

    # Saving the best model
    trainer.save_model(results_dir / "model")
    mlflow.log_artifacts(results_dir / "model", artifact_path="model")

if __name__ == '__main__':
    main()
