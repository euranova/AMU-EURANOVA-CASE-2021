"""This file contains the code to evaluate a token classification task under the BIO format."""
from __future__ import division, print_function, unicode_literals

"""
This script applies to IOB2 or IOBES tagging scheme.
If you are using a different scheme, please convert to IOB2 or IOBES.
IOB2:
- B = begin,
- I = inside but not the first,
- O = outside
e.g.
John   lives in New   York  City  .
B-PER  O     O  B-LOC I-LOC I-LOC O
IOBES:
- B = begin,
- E = end,
- S = singleton,
- I = inside but not the first or the last,
- O = outside
e.g.
John   lives in New   York  City  .
S-PER  O     O  B-LOC I-LOC E-LOC O
prefix: IOBES
chunk_type: PER, LOC, etc.
"""
from collections import defaultdict
import itertools
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


class EvalTokenClassif():
    def __init__(self, result_folder='results/task_3'):
        self._result_folder = Path(result_folder)
        self._result_folder.mkdir(parents=True, exist_ok=True)

    def _split_tag(self, chunk_tag):
        """
        split chunk tag into IOBES prefix and chunk_type
        e.g.
        B-PER -> (B, PER)
        O -> (O, None)
        """
        if chunk_tag == 'O':
            return ('O', None)
        return chunk_tag.split('-', maxsplit=1)

    def _is_chunk_end(self, prev_tag, tag):
        """
        check if the previous chunk ended between the previous and current word
        e.g.
        (B-PER, I-PER) -> False
        (B-LOC, O)  -> True
        Note: in case of contradicting tags, e.g. (B-PER, I-LOC)
        this is considered as (B-PER, B-LOC)
        """
        prefix1, chunk_type1 = self._split_tag(prev_tag)
        prefix2, chunk_type2 = self._split_tag(tag)

        if prefix1 == 'O':
            return False
        if prefix2 == 'O':
            return prefix1 != 'O'

        if chunk_type1 != chunk_type2:
            return True

        return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']

    def _is_chunk_start(self, prev_tag, tag):
        """
        check if a new chunk started between the previous and current word
        """
        prefix1, chunk_type1 = self._split_tag(prev_tag)
        prefix2, chunk_type2 = self._split_tag(tag)

        if prefix2 == 'O':
            return False
        if prefix1 == 'O':
            return prefix2 != 'O'

        if chunk_type1 != chunk_type2:
            return True

        return prefix2 in ['B', 'S'] or prefix1 in ['E', 'S']


    def _calc_metrics(self, tp, p, t, percent=True):
        """
        compute overall precision, recall and FB1 (default values are 0.0)
        if percent is True, return 100 * original decimal value
        """
        precision = tp / p if p else 0
        recall = tp / t if t else 0
        fb1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        if percent:
            return 100 * precision, 100 * recall, 100 * fb1
        else:
            return precision, recall, fb1


    def _count_chunks(self, true_seqs, pred_seqs):
        """
        true_seqs: a list of true tags
        pred_seqs: a list of predicted tags
        return:
        correct_chunks: a dict (counter),
                        key = chunk types,
                        value = number of correctly identified chunks per type
        true_chunks:    a dict, number of true chunks per type
        pred_chunks:    a dict, number of identified chunks per type
        correct_counts, true_counts, pred_counts: similar to above, but for tags
        """
        correct_chunks = defaultdict(int)
        true_chunks = defaultdict(int)
        pred_chunks = defaultdict(int)

        correct_counts = defaultdict(int)
        true_counts = defaultdict(int)
        pred_counts = defaultdict(int)

        prev_true_tag, prev_pred_tag = 'O', 'O'
        correct_chunk = None

        for true_tag, pred_tag in zip(true_seqs, pred_seqs):
            if true_tag == pred_tag:
                correct_counts[true_tag] += 1
            true_counts[true_tag] += 1
            pred_counts[pred_tag] += 1

            _, true_type = self._split_tag(true_tag)
            _, pred_type = self._split_tag(pred_tag)

            if correct_chunk is not None:
                true_end = self._is_chunk_end(prev_true_tag, true_tag)
                pred_end = self._is_chunk_end(prev_pred_tag, pred_tag)

                if pred_end and true_end:
                    correct_chunks[correct_chunk] += 1
                    correct_chunk = None
                elif pred_end != true_end or true_type != pred_type:
                    correct_chunk = None

            true_start = self._is_chunk_start(prev_true_tag, true_tag)
            pred_start = self._is_chunk_start(prev_pred_tag, pred_tag)

            if true_start and pred_start and true_type == pred_type:
                correct_chunk = true_type
            if true_start:
                true_chunks[true_type] += 1
            if pred_start:
                pred_chunks[pred_type] += 1

            prev_true_tag, prev_pred_tag = true_tag, pred_tag
        if correct_chunk is not None:
            correct_chunks[correct_chunk] += 1

        return (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts)

    def _get_result(self, correct_chunks, true_chunks, pred_chunks,
        correct_counts, true_counts, pred_counts, verbose=True):
        """
        if verbose, print overall performance, as well as preformance per chunk type;
        otherwise, simply return overall prec, rec, f1 scores
        """
        # sum counts
        sum_correct_chunks = sum(correct_chunks.values())
        sum_true_chunks = sum(true_chunks.values())
        sum_pred_chunks = sum(pred_chunks.values())

        sum_correct_counts = sum(correct_counts.values())
        sum_true_counts = sum(true_counts.values())

        nonO_correct_counts = sum(v for k, v in correct_counts.items() if k != 'O')
        nonO_true_counts = sum(v for k, v in true_counts.items() if k != 'O')

        chunk_types = sorted(list(set(list(true_chunks) + list(pred_chunks))))

        # compute overall precision, recall and FB1 (default values are 0.0)
        prec, rec, f1 = self._calc_metrics(sum_correct_chunks, sum_pred_chunks, sum_true_chunks)
        res = (prec, rec, f1)
        if not verbose:
            return res

        # print overall performance, and performance per chunk type

        results_file_path = self._result_folder / 'score_results.txt'

        with results_file_path.open('w+') as results_file:
            results_file.write("processed %i tokens with %i phrases; " % (sum_true_counts, sum_true_chunks))

            results_file.write("found: %i phrases; correct: %i.\n" % (sum_pred_chunks, sum_correct_chunks))

            results_file.write("accuracy: %6.2f%%; (non-O)" % (100*nonO_correct_counts/nonO_true_counts))
            results_file.write("accuracy: %6.2f%%; " % (100*sum_correct_counts/sum_true_counts))
            results_file.write("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" % (prec, rec, f1))

            # for each chunk type, compute precision, recall and FB1 (default values are 0.0)
            for t in chunk_types:
                prec, rec, f1 = self._calc_metrics(correct_chunks[t], pred_chunks[t], true_chunks[t])
                results_file.write("%17s: " %t)
                results_file.write("precision: %6.2f%%; recall: %6.2f%%; FB1: %6.2f" %
                        (prec, rec, f1))
                results_file.write("  %d" % pred_chunks[t])

        return res
        # you can generate LaTeX output for tables like in
        # http://cnts.uia.ac.be/conll2003/ner/example.tex
        # but I'm not implementing this

    def _evaluate(self, true_seqs, pred_seqs, verbose=True):
        (correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts) = self._count_chunks(true_seqs, pred_seqs)
        result = self._get_result(correct_chunks, true_chunks, pred_chunks,
            correct_counts, true_counts, pred_counts, verbose=verbose)
        return result

    def _get_data(self, testf):

        label_per_tag = []
        label_per_class = []

        with open(testf) as f:
            for line in f.readlines():
                line_stripped = line.strip()
                line_splitted = line_stripped.split("\t")
                if len(line_splitted) > 1:
                    label_per_tag.append(line_splitted[1])
                    ## ONLY FOR TASK 3 ##
                    if line_splitted[1] != "O":
                        label_per_class.append(line_splitted[1].split("-")[1])
                    else:
                        label_per_class.append(line_splitted[1])

        return label_per_tag, label_per_class

    def _plot_confusion_matrix(self, data_gold_plot, data_predicted_plot, data_gold_eval, data_predicted_eval):
        """
        plot confusion matrix
        """
        # CM per CLASS - BIO
        gold_array1 = np.asarray(data_gold_eval)
        predicted_array2 = np.asarray(data_predicted_eval)

        cont_full_1 = pd.crosstab(gold_array1,predicted_array2, normalize='index').round(3)*100

        return cont_full_1

    def _plot_heatmaps(self, cont_full_1):
        """
        PLOT with full labels # w. BIO
        """
        plt.subplots(figsize=(6, 6))
        sns.heatmap(cont_full_1, annot=True, fmt='g', linewidths=.5, cmap="YlGnBu")
        plt.xticks(rotation=70)
        
        results_file_path = self._result_folder / 'score_plot.png'

        plt.savefig(results_file_path)


    def evaluate(self, gold_data, predicted_data):
        """
        Script modified to take 2 files in input and run evaluation.

        usage: python3 conlleval-python_bestCLEF.py [gold-file] [predicted-file]
        """
        data_gold_eval, data_gold_plot = self._get_data(gold_data)
        data_predicted_eval, data_predicted_plot= self._get_data(predicted_data)

        ## ONLY FOR TASK 3 ##
        results = self._evaluate(data_gold_eval, data_predicted_eval)

        cont_full = self._plot_confusion_matrix(data_gold_plot, data_predicted_plot, data_gold_eval, data_predicted_eval)

        self._plot_heatmaps(cont_full)

        return results
