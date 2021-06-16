"""Callback for token classification to compute metric on train too."""
import collections
import time
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data.dataset import Dataset
from transformers import Trainer
from transformers.trainer_utils import speed_metrics
from transformers.file_utils import is_torch_tpu_available

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

from event_extraction.model.loss_derivable_f1 import loss_macro_f1, loss_micro_f1

class TrainerTokenClassif(Trainer):
    """Specific trainer to add train metric to logging and custom loss.
    
    Code from: https://discuss.huggingface.co/t/logging-training-accuracy-using-trainer-class/5524
    """

    def __init__(self, model=None, args = None, data_collator = None, train_dataset = None, 
             eval_dataset = None, tokenizer = None, model_init = None, compute_metrics = None, 
             callbacks = None, optimizers = (None,None), loss="macro"):
        """Init function of specific Trainer class.

        Args:
            model ([type]): [description]
            args ([type], optional): [description]. Defaults to None.
            data_collator ([type], optional): [description]. Defaults to None.
            train_dataset ([type], optional): [description]. Defaults to None.
            eval_dataset ([type], optional): [description]. Defaults to None.
            tokenizer ([type], optional): [description]. Defaults to None.
            model_init ([type], optional): [description]. Defaults to None.
            compute_metrics ([type], optional): [description]. Defaults to None.
            callbacks ([type], optional): [description]. Defaults to None.
            optimizers (tuple, optional): [description]. Defaults to (None,None).
            loss(str, optional): base, macro or micro. Defaults to macro.
        """
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init,
                         compute_metrics, callbacks, optimizers)
        self._loss = loss

    def evaluate(self, train_dataset = None, eval_dataset: Optional[Dataset] = None, 
                 ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval",) -> Dict[str, float]:
        """Evaluate method.

        Args:
            train_dataset ([type], optional): [description]. Defaults to None.
            eval_dataset (Optional[Dataset], optional): [description]. Defaults to None.
            ignore_keys (Optional[List[str]], optional): [description]. Defaults to None.
            metric_key_prefix (str, optional): [description]. Defaults to "eval".

        Raises:
            ValueError: [description]

        Returns:
            Dict[str, float]: [description]
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")
        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        train_output = self.prediction_loop(
            train_dataloader,
            description = 'Training',
            prediction_loss_only = True if self.compute_metrics is None else None,
            ignore_keys = ignore_keys,
            metric_key_prefix = 'train',
            )


        eval_output = self.prediction_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        train_n_samples = len(self.train_dataset)
        train_output.metrics.update(speed_metrics('train', start_time, train_n_samples))
        self.log(train_output.metrics)

        eval_n_samples = len(eval_dataset if eval_dataset is not None else self.eval_dataset)
        eval_output.metrics.update(speed_metrics(metric_key_prefix, start_time, eval_n_samples))
        self.log(eval_output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        # Hackitty hack from the link in the docstring
        eval_output.metrics["eval_loss"] = "No log"
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, eval_output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, train_output.metrics)

        self._memory_tracker.stop_and_update_metrics(train_output.metrics)
        self._memory_tracker.stop_and_update_metrics(eval_output.metrics)

        dic = {
        'Training metrics': train_output.metrics,
        'Validation metrics': eval_output.metrics,
        'eval_f1': eval_output.metrics['eval_f1']
        }

        return dic
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss chosen in the init.

        Args:
            model ([type]): [description]
            inputs ([type]): [description]
            return_outputs (bool, optional): [description]. Defaults to False.

        Returns:
            [type]: [description]
        """
        if self._loss == "base":
            output_loss = super.compute_loss(model, inputs, return_outputs)
        else:
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            labels = labels.view(-1).cpu()

            outputs_logits = torch.nn.functional.softmax(outputs.logits, dim=2)
            outputs_logits = torch.flatten(outputs_logits, end_dim=1).cpu()

            # Mask out 'PAD' tokens
            mask = (labels >= 0).float().cpu()
            if self._loss == "micro":
                loss = loss_micro_f1(labels, outputs_logits, mask)
            elif self._loss == "macro":
                loss = loss_macro_f1(labels, outputs_logits, mask)
            else:
                raise ValueError("loss must be base, micro or macro.")

            output_loss = (loss, outputs) if return_outputs else loss
        return output_loss

    def hyperparameter_search(self,
        hp_space: Optional[Callable[["optuna.Trial"], Dict[str, float]]] = None,
        compute_objective: Optional[Callable[[Dict[str, float]], float]] = None,
        n_trials: int = 20,
        direction: str = "minimize",
        backend: Optional[Union["str", HPSearchBackend]] = None,
        hp_name: Optional[Callable[["optuna.Trial"], str]] = None,
        **kwargs,
    ) -> BestRun:
        """
        Launch an hyperparameter search using ``optuna`` or ``Ray Tune``. The optimized quantity is determined by
        :obj:`compute_objective`, which defaults to a function returning the evaluation loss when no metric is
        provided, the sum of all metrics otherwise.

        .. warning::

            To use this method, you need to have provided a ``model_init`` when initializing your
            :class:`~transformers.Trainer`: we need to reinitialize the model at each new run. This is incompatible
            with the ``optimizers`` argument, so you need to subclass :class:`~transformers.Trainer` and override the
            method :meth:`~transformers.Trainer.create_optimizer_and_scheduler` for custom optimizer/scheduler.

        Args:
            hp_space (:obj:`Callable[["optuna.Trial"], Dict[str, float]]`, `optional`):
                A function that defines the hyperparameter search space. Will default to
                :func:`~transformers.trainer_utils.default_hp_space_optuna` or
                :func:`~transformers.trainer_utils.default_hp_space_ray` depending on your backend.
            compute_objective (:obj:`Callable[[Dict[str, float]], float]`, `optional`):
                A function computing the objective to minimize or maximize from the metrics returned by the
                :obj:`evaluate` method. Will default to :func:`~transformers.trainer_utils.default_compute_objective`.
            n_trials (:obj:`int`, `optional`, defaults to 100):
                The number of trial runs to test.
            direction(:obj:`str`, `optional`, defaults to :obj:`"minimize"`):
                Whether to optimize greater or lower objects. Can be :obj:`"minimize"` or :obj:`"maximize"`, you should
                pick :obj:`"minimize"` when optimizing the validation loss, :obj:`"maximize"` when optimizing one or
                several metrics.
            backend(:obj:`str` or :class:`~transformers.training_utils.HPSearchBackend`, `optional`):
                The backend to use for hyperparameter search. Will default to optuna or Ray Tune, depending on which
                one is installed. If both are installed, will default to optuna.
            kwargs:
                Additional keyword arguments passed along to :obj:`optuna.create_study` or :obj:`ray.tune.run`. For
                more information see:

                - the documentation of `optuna.create_study
                    <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.create_study.html>`__
                - the documentation of `tune.run
                    <https://docs.ray.io/en/latest/tune/api_docs/execution.html#tune-run>`__

        Returns:
            :class:`transformers.trainer_utils.BestRun`: All the information about the best run.
        """
        ray.init()
        tune.utils.wait_for_gpu()
        best_run = super.hyperparameter_search(hp_space, compute_objective, n_trials, direction,
                                               backend, hp_name, **kwargs)
        return best_run
