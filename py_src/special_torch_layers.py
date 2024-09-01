__ignore_layer_list_averaging = ["num_batches_tracked", "running_mean", "running_var"]

def is_ignored_layer_averaging(layer_name):
    return is_keyword_in_layer_name(layer_name, __ignore_layer_list_averaging)


__ignore_layer_list_variance_correction = ["num_batches_tracked", "running_mean", "running_var"]

def is_ignored_layer_variance_correction(layer_name):
    return is_keyword_in_layer_name(layer_name, __ignore_layer_list_variance_correction)


def is_keyword_in_layer_name(layer_name, keywords):
    output = False
    for i in keywords:
        if i in layer_name:
            output = True
            break
    return output


__normalization_layer_layer_keyword = {'lenet5': None, 'resnet18_bn': ['bn'], 'resnet18_gn': ['bn']}
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
