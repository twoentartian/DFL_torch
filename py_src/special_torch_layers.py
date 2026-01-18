import re
import torch.nn as nn

def is_keyword_in_layer_name(layer_name, keywords):
    output = False
    for i in keywords:
        if i in layer_name:
            output = True
            break
    return output

def is_layer_index_with_keyword(layer_name, keywords):
    output = False
    for i in keywords:
        m = re.search(r'^[^.]+\.(\d+)$', layer_name)
        layer_index = int(m.group(1)) if m else None
        m = re.search(r'^[^.]+\.(\d+)$', i)
        layer_index_in_keyword = int(m.group(1)) if m else None
        if layer_name.startswith(i) and layer_index == layer_index_in_keyword:
            output = True
            break
    return output

""" ignore averaging """
__ignore_layer_list_averaging = ["num_batches_tracked", "running_mean", "running_var"]
def is_ignored_layer_averaging(layer_name):
    return is_keyword_in_layer_name(layer_name, __ignore_layer_list_averaging)

""" ignore variance correction """
__ignore_layer_list_variance_correction = ["num_batches_tracked", "running_mean", "running_var"]
def is_ignored_layer_variance_correction(layer_name):
    return is_keyword_in_layer_name(layer_name, __ignore_layer_list_variance_correction)


__ignore_layer_list_normalization = ["num_batches_tracked", "running_mean", "running_var"]
__normalization_layer_layer_keyword = {'lenet5': None, 'resnet18_bn': ['bn'], 'resnet18_gn': ['bn'], 'cct7':['classifier.norm']}
def is_normalization_layer(model_name, layer_name):
    output = False
    if model_name not in __normalization_layer_layer_keyword.keys():
        raise NotImplementedError(f"{model_name} not found in NORMALIZATION_LAYER_KEYWORD: {__normalization_layer_layer_keyword}")
    keywords = __normalization_layer_layer_keyword[model_name]
    if keywords is None:
        return False
    for i in keywords:
        if i in layer_name:
            output = True
            break
    return output




def find_layers_according_to_name_and_keyword(model_state_dict, layer_names=None, layer_name_keywords=None, layer_name_match_layer_index=None):
    found_layers = []
    ignored_layers = []
    if layer_names is None:
        _layer_names = []
    else:
        _layer_names = layer_names
    if layer_name_keywords is None:
        _layer_name_keywords = []
    else:
        _layer_name_keywords = layer_name_keywords
    if layer_name_match_layer_index is None:
        _layer_name_start_with_keyword = []
    else:
        _layer_name_start_with_keyword = layer_name_match_layer_index
    for l in model_state_dict.keys():
        if l in _layer_names:
            found_layers.append(l)
        if is_keyword_in_layer_name(l, _layer_name_keywords):
            found_layers.append(l)
        if is_layer_index_with_keyword(l, _layer_name_start_with_keyword):
            found_layers.append(l)
    for l in model_state_dict.keys():
        if l not in found_layers:
            ignored_layers.append(l)
    return found_layers, ignored_layers


class normalization_layer_results:
    def __init__(self):
        self.batch_normalization = []
        self.layer_normalization = []
        self.group_normalization = []
        self.instance_normalization = []

def find_normalization_layers(model):
    output = normalization_layer_results()
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,)):
            output.batch_normalization.append(name)
        if isinstance(module, (nn.LayerNorm)):
            output.layer_normalization.append(name)
        if isinstance(module, (nn.GroupNorm)):
            output.group_normalization.append(name)
        if isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,)):
            output.instance_normalization.append(name)
    return output