__ignore_layer_list = ["num_batches_tracked", "running_mean", "running_var"]


def is_ignored_layer(layer_name):
    output = False
    for i in __ignore_layer_list:
        if i in layer_name:
            output = True
            break
    return output
