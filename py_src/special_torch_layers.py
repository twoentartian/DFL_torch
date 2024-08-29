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

