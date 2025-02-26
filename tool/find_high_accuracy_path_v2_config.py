
from find_high_accuracy_path_v2.find_parameters import ParameterMove, ParameterTrain, ParameterRebuildNorm
from find_high_accuracy_path_v2.runtime_parameters import RuntimeParameters

max_tick = 10000

def get_parameter_move(runtime_parameter: RuntimeParameters):
    output = ParameterMove()
    output.set_default()

    return output


def get_parameter_train(runtime_parameter: RuntimeParameters):
    output = ParameterTrain()
    output.set_default()

    return output

def get_parameter_rebuild_norm(runtime_parameter: RuntimeParameters):
    output = ParameterRebuildNorm()
    output.set_default()

    return output
