"""Test the hyperparameter search."""

from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.suggest import ConcurrencyLimiter
from transformers import TrainingArguments

from event_extraction.model.bert_token_classif import BertTokenClassif

def test_hyperparameter_local_model():
    """Test hyperparameter search on a model saved locally."""
    texts = [["test"]*600, ["Kilimandjaro"]*300, ["je", "suis", "un", "test"], ["un"]+["unun"]*600]
    labels = [["O"]*600, ["O"]*300, ["B", "I", "I", "I"], ["O"]*601]

    training_arguments = TrainingArguments(output_dir="tests/results_test", num_train_epochs=1, save_strategy="no")

    trainer = BertTokenClassif(pretrained_model="./tests/test_data/dummy_model/",
                               training_arguments=training_arguments)

    # Defining elements of ray.
    hp_space = {
        "num_train_epochs":tune.choice([2, 3, 4]),
        "weight_decay": tune.uniform(0.0001, 1),
        "learning_rate": tune.choice([1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 2e-7, 1e-7, 3e-7, 2e-8]),
        "adafactor": tune.choice(['True','False']),
        "adam_beta1": tune.uniform(0.0, 1.0),
        "adam_beta2": tune.uniform(0, 1.0),
        "adam_epsilon": tune.choice([1e-8, 2e-8, 3e-8, 1e-9, 2e-9, 3e-10]),
        "max_grad_norm": tune.uniform(0, 1.0),
    }
    resources_per_trial={
        "cpu":1,
    }
    search_alg = HyperOptSearch(metric="eval_f1", mode="max", n_initial_points=5)
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1, batch=True)

    def get_eval_f1(metrics):
        """Return the eval_F1, useful for hyperparameter search.

        Args:
            metrics (Dict): dictionnary with all the metrics.

        Returns:
            float: the F1 on the evaluation dataset.
        """
        return metrics["eval_f1"]

    best_run = trainer.hyperparameter_search(texts, labels, texts, labels, hp_space=hp_space,
                                             n_trials=2,direction="maximize", backend="ray", search_alg=search_alg,
                                             compute_objective=get_eval_f1, resources_per_trial=resources_per_trial,
                                            )
