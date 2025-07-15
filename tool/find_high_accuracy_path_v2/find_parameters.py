class Parameter(object):
    def validate(self):
        assert all(value is not None for value in vars(self).values())

class ParameterGeneral(Parameter):
    max_tick = None
    dataloader_worker = None
    test_dataset_use_whole = None   # False by default

class ParameterMove(Parameter):
    step_size = None
    adoptive_step_size = None
    layer_skip_move = None
    layer_skip_move_keyword = None
    merge_bias_with_weights = None

    # layer norm layers
    layer_compensate_x2 = None
    layer_compensate_x2_keyword = None

    # attention layers
    layer_attention = None
    layer_attention_keyword = None
    layer_attention_policy = None # 'none', 'ignore_kv'

    def fill_default(self):
        if self.layer_compensate_x2 is None:
            self.layer_compensate_x2 = []
        if self.layer_compensate_x2_keyword is None:
            self.layer_compensate_x2_keyword = []
        if self.layer_attention is None:
            self.layer_attention = []
        if self.layer_attention_keyword is None:
            self.layer_attention_keyword = []
        if self.layer_attention_policy is None:
            self.layer_attention_policy = 'none'

class ParameterTrain(Parameter):
    train_for_max_rounds = None
    train_until_loss = None
    train_for_min_rounds = None
    pretrain_optimizer = None
    load_existing_optimizer = None

    def fill_default(self):
        pass

class ParameterRebuildNorm(Parameter):
    rebuild_norm_for_max_rounds = None
    rebuild_norm_for_min_rounds = None
    rebuild_norm_until_loss = None
    rebuild_norm_layer = None
    rebuild_norm_layer_keyword = None

    rebuild_norm_use_initial_norm_weights = False
    rebuild_norm_use_start_model_norm_weights = False