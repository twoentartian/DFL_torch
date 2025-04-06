import torch
import sys
import os

from find_high_accuracy_path_v2.find_parameters import ParameterMove, ParameterTrain, ParameterRebuildNorm, ParameterGeneral
from find_high_accuracy_path_v2.runtime_parameters import RuntimeParameters

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src.ml_setup import MlSetup

model_name = 'cct7'

def get_parameter_general(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterGeneral()
    if ml_setup.model_name == model_name:
        output.max_tick = 40000
        output.dataloader_worker = 8
        output.test_dataset_use_whole = True
    else:
        raise NotImplemented
    return output

def get_parameter_move(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterMove()
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["classifier.blocks."]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 5000:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["classifier.blocks.1", "classifier.blocks.2","classifier.blocks.3","classifier.blocks.4","classifier.blocks.5","classifier.blocks.6"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 10000:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["classifier.blocks.2","classifier.blocks.3","classifier.blocks.4","classifier.blocks.5","classifier.blocks.6"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 15000:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["classifier.blocks.3","classifier.blocks.4","classifier.blocks.5","classifier.blocks.6"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 20000:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["classifier.blocks.4","classifier.blocks.5","classifier.blocks.6"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 25000:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["classifier.blocks.5","classifier.blocks.6"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 30000:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = ["classifier.blocks.6"]
            output.merge_bias_with_weights = False
        elif runtime_parameter.current_tick == 35000:
            output.step_size = 0.001
            output.adoptive_step_size = 0.001
            output.layer_skip_move = []
            output.layer_skip_move_keyword = []
            output.merge_bias_with_weights = False
        else:
            return None
    else:
        raise NotImplemented
    return output


def get_parameter_train(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterTrain()
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            output.train_for_max_rounds = 10000
            output.train_for_min_rounds = 10
            output.train_until_loss = 0.05
            output.pretrain_optimizer = False
            output.load_existing_optimizer = False
        else:
            return None
    else:
        raise NotImplemented
    return output

def get_optimizer_train(runtime_parameter: RuntimeParameters, ml_setup: MlSetup, model_parameter):
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            optimizer = torch.optim.SGD(model_parameter, lr=0.001)       # LeNet5
        else:
            return None
    else:
        raise NotImplemented
    return optimizer

def get_parameter_rebuild_norm(runtime_parameter: RuntimeParameters, ml_setup: MlSetup):
    output = ParameterRebuildNorm()
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            output.rebuild_norm_for_max_rounds = 0
            output.rebuild_norm_for_min_rounds = 0
            output.rebuild_norm_until_loss = 10
            output.rebuild_norm_layer = []
            output.rebuild_norm_layer_keyword = []
        else:
            return None
    else:
        raise NotImplemented
    return output

def get_optimizer_rebuild_norm(runtime_parameter: RuntimeParameters, ml_setup: MlSetup, model_parameter):
    if ml_setup.model_name == model_name:
        if runtime_parameter.current_tick == 0:
            optimizer = None
        else:
            return None
    else:
        raise NotImplemented
    return optimizer