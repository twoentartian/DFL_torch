import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from py_src.ml_setup_base.base import MlSetup, TrainStepOutput
import py_src.ml_setup_base.dataset as ml_setup_dataset

from py_src.ml_setup_base.model import ModelType
from py_src.ml_setup_base import dataset_modular, transformer_for_grokking

def step(batch_index, batch, model: transformer_for_grokking.Transformer, optimizer: torch.optim.Optimizer, lr_scheduler, arg_ml_setup: MlSetup, train=False):
    if train:
        optimizer.zero_grad(set_to_none=True)

    x = batch["text"]
    y = batch["target"]
    y_hat, attentions, values = model.forward(x=x)
    y_hat = y_hat.transpose(-2, -1)

    tokenizer = arg_ml_setup.training_data.tokenizer
    eq_token_index = tokenizer.stoi["="]
    eq_position_t = torch.nonzero(y[0, :] == eq_token_index, as_tuple=False)
    eq_position = int(eq_position_t.squeeze())
    y_rhs = y[..., eq_position + 1:]
    y_hat_rhs = y_hat[..., eq_position + 1:]
    x_lhs = x[..., : eq_position + 1]
    loss = F.cross_entropy(y_hat_rhs, y_rhs, reduction="mean")

    # find max prediction from output
    y_hat_max = torch.max(y_hat_rhs, dim=-2).indices  # batchsize x num_rhs_tokens
    row_accuracy = torch.min((y_hat_max == y_rhs), dim=-1).values  # shape: batchsize
    correct_count = row_accuracy.int().sum()

    if train:
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

    output = TrainStepOutput()
    output.loss_value = loss.item()
    output.sample_count = y.shape[0]
    output.correct_count = correct_count.item()
    return output

def train_step(batch_index, batch, model: transformer_for_grokking.Transformer, optimizer: torch.optim.Optimizer, lr_scheduler, arg_ml_setup: MlSetup) -> TrainStepOutput:
    return step(batch_index, batch, model, optimizer, lr_scheduler, arg_ml_setup, train=True)

def evaluation_step(batch_index, batch, model: transformer_for_grokking.Transformer, optimizer: torch.optim.Optimizer, lr_scheduler, arg_ml_setup: MlSetup) -> TrainStepOutput:
    return step(batch_index, batch, model, optimizer, lr_scheduler, arg_ml_setup, train=False)

def arithmetic_addition_grokking(device, train_percentage: float=50, operand_length: Optional[int]=None):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_arithmetic_addition(train_percentage=train_percentage, operand_length=operand_length)
    output_ml_setup.model = transformer_for_grokking.Transformer(n_layers=2, n_heads=4, d_model=128, max_context_len=50)
    output_ml_setup.model_name = str(ModelType.transformer_for_grokking.name)
    output_ml_setup.model_type = ModelType.transformer_for_grokking
    output_ml_setup.get_info_from_dataset(dataset)

    output_ml_setup.training_batch_size = min(512, math.ceil(len(dataset.training_data) / 2.0))
    output_ml_setup.has_normalization_layer = True

    output_ml_setup.override_training_dataset_loader = dataset_modular.ArithmeticIterator(dataset.training_data, device, batchsize_hint=-1)
    output_ml_setup.override_testing_dataset_loader = dataset_modular.ArithmeticIterator(dataset.testing_data, device, batchsize_hint=-1)
    output_ml_setup.criterion = nn.CrossEntropyLoss()

    output_ml_setup.override_train_step_function = train_step
    output_ml_setup.override_evaluation_step_function = evaluation_step

    return output_ml_setup

def arithmetic_cubepoly_grokking(device, train_percentage: float=50, operand_length: Optional[int]=None):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_arithmetic_cubepoly(train_percentage=train_percentage, operand_length=operand_length)
    output_ml_setup.model = transformer_for_grokking.Transformer(n_layers=2, n_heads=4, d_model=128, max_context_len=50)
    output_ml_setup.model_name = str(ModelType.transformer_for_grokking.name)
    output_ml_setup.model_type = ModelType.transformer_for_grokking
    output_ml_setup.get_info_from_dataset(dataset)

    output_ml_setup.training_batch_size = min(512, math.ceil(len(dataset.training_data) / 2.0))
    output_ml_setup.has_normalization_layer = True

    output_ml_setup.override_training_dataset_loader = dataset_modular.ArithmeticIterator(dataset.training_data, device, batchsize_hint=-1)
    output_ml_setup.override_testing_dataset_loader = dataset_modular.ArithmeticIterator(dataset.testing_data, device, batchsize_hint=-1)
    output_ml_setup.criterion = nn.CrossEntropyLoss()

    output_ml_setup.override_train_step_function = train_step
    output_ml_setup.override_evaluation_step_function = evaluation_step

    return output_ml_setup

def arithmetic_cube2_grokking(device, train_percentage: float=50, operand_length: Optional[int]=None):
    output_ml_setup = MlSetup()
    dataset = ml_setup_dataset.dataset_arithmetic_cube2(train_percentage=train_percentage, operand_length=operand_length)
    output_ml_setup.model = transformer_for_grokking.Transformer(n_layers=2, n_heads=4, d_model=128, max_context_len=50)
    output_ml_setup.model_name = str(ModelType.transformer_for_grokking.name)
    output_ml_setup.model_type = ModelType.transformer_for_grokking
    output_ml_setup.get_info_from_dataset(dataset)

    output_ml_setup.training_batch_size = min(512, math.ceil(len(dataset.training_data) / 2.0))
    output_ml_setup.has_normalization_layer = True

    output_ml_setup.override_training_dataset_loader = dataset_modular.ArithmeticIterator(dataset.training_data, device, batchsize_hint=-1)
    output_ml_setup.override_testing_dataset_loader = dataset_modular.ArithmeticIterator(dataset.testing_data, device, batchsize_hint=-1)
    output_ml_setup.criterion = nn.CrossEntropyLoss()

    output_ml_setup.override_train_step_function = train_step
    output_ml_setup.override_evaluation_step_function = evaluation_step

    return output_ml_setup
