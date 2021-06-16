"""Test the custom loss to verify if it's acting the way expected."""

import torch
from transformers import AutoModelForTokenClassification

from event_extraction.model.loss_derivable_f1 import *


def test_f1_macro_loss():
    """Test the F1 macro loss."""
    y_true = torch.tensor([0, 1, 2, 2])
    y_pred_true = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
    y_pred_false = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [1., 0., 0.]])
    y_pred_medium = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.], [1., 0., 0.]])
    no_mask = torch.tensor([1., 1., 1., 1.])
    mask = torch.tensor([1., 1., 0., 1.])

    assert torch.isclose(
        loss_macro_f1(y_true, y_pred_true, no_mask), torch.tensor(0.), atol=1e-06)
    assert torch.isclose(
        loss_macro_f1(y_true, y_pred_false, no_mask), torch.tensor(1.), atol=1e-06)
    assert torch.isclose(
        loss_macro_f1(y_true, y_pred_medium, no_mask), torch.tensor(1-(7/6)/3))
    assert torch.isclose(
        loss_macro_f1(y_true, y_pred_true, mask), torch.tensor(0.), atol=1e-06)
    assert torch.isclose(
        loss_macro_f1(y_true, y_pred_false, mask), torch.tensor(1.), atol=1e-06)
    assert torch.isclose(
        loss_macro_f1(y_true, y_pred_medium, mask), torch.tensor((2/3*2+1)/3), atol=1e-06)

def test_f1_micro_loss():
    """Test the F1 micro loss."""
    y_true = torch.tensor([0, 1, 2, 2])
    y_pred_true = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
    y_pred_false = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [1., 0., 0.]])
    y_pred_medium = torch.tensor([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.], [1., 0., 0.]])
    no_mask = torch.tensor([1., 1., 1., 1.])
    mask = torch.tensor([1., 0., 1., 0.])

    assert torch.isclose(
        loss_micro_f1(y_true, y_pred_true, no_mask), torch.tensor(0.))
    assert torch.isclose(
        loss_micro_f1(y_true, y_pred_false, no_mask), torch.tensor(1.))
    assert torch.isclose(
        loss_micro_f1(y_true, y_pred_medium, no_mask), torch.tensor(1/2))
    assert torch.isclose(
        loss_micro_f1(y_true, y_pred_true, mask), torch.tensor(0.))
    assert torch.isclose(
        loss_micro_f1(y_true, y_pred_false, mask), torch.tensor(1.))
    assert torch.isclose(
        loss_micro_f1(y_true, y_pred_medium, mask), torch.tensor(0.))
