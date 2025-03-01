class Parameter(object):
    def validate(self):
        assert all(value is not None for value in vars(self).values())

class ParameterGeneral(Parameter):
    max_tick = None
    dataloader_worker = None
    test_dataset_use_whole = None

class ParameterMove(Parameter):
    step_size = None
    adoptive_step_size = None
    layer_skip_move = None
    layer_skip_move_keyword = None
    merge_bias_with_weights = None

class ParameterTrain(Parameter):
    train_for_max_rounds = None
    train_until_loss = None
    train_for_min_rounds = None
    pretrain_optimizer = None
    load_existing_optimizer = None

class ParameterRebuildNorm(Parameter):
    rebuild_norm_for_max_rounds = None
    rebuild_norm_for_min_rounds = None
    rebuild_norm_until_loss = None
    rebuild_norm_layer = None
    rebuild_norm_layer_keyword = None