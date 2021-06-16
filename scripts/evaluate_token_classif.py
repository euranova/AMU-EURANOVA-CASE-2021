"""Evaluate a token classification task."""

import click

from event_extraction.statistic.eval_token_classif import EvalTokenClassif

@click.command()
@click.option("--predict_file", help="File the predictions.")
@click.option("--gold_file", help="File with the gold standard.")
@click.option("--result_folder", help="Folder to put the results of the evaluation.")
def main(predict_file, gold_file, result_folder):
    """Compute results for token classification task.

    Args:
        predict_file (str): file the predictions.
        gold_file (str): file with the gold standard.
        result_folder (str): folder to put the results of the evaluation.
    """
    evaluator = EvalTokenClassif(result_folder=result_folder)

    evaluator.evaluate(gold_file, predict_file)


if __name__ == "__main__":
    main()
