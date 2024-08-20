__ignore_layer_list_averaging = ["num_batches_tracked", "running_mean", "running_var"]

def is_ignored_layer_averaging(layer_name):
    output = False
    for i in __ignore_layer_list_averaging:
        if i in layer_name:
            output = True
            break
    return output


__ignore_layer_list_variance_correction = ["num_batches_tracked", "running_mean", "running_var"]

def is_ignored_layer_variance_correction(layer_name):
    output = False
    for i in __ignore_layer_list_variance_correction:
        if i in layer_name:
            output = True
            break
    return output
