from enum import Enum, auto

class WorkMode(Enum):
    unknown = auto()
    to_origin = auto()
    to_inf = auto()
    to_certain_model = auto()
    to_mean = auto()
    to_vs = auto()

class RuntimeParameters(object):
    start_and_end_point_for_paths = None

    """should not change these values during simulation"""
    use_cpu = None
    use_amp = None
    work_mode = WorkMode.unknown
    output_folder_path = None
    total_cpu_count = None
    worker_count = None

    save_ticks = None
    save_interval = None
    save_format = None

    config_file_path = None
    dataset_name = None
    model_name = None
    pytorch_preset_version = None
    store_top_accuracy_model_count = None
    checkpoint_interval = None
    task_name = None
    silence_mode = None
    across_vs_lr_policy = None
    linear_interpolation_points_size = None
    linear_interpolation_dataset_size = None
    variance_sphere_file_path = None
    variance_sphere_model = None

    """real-time values"""
    current_tick = None
    max_tick = None

    debug_check_config_mode = None
    test_dataset_use_whole = None
    service_test_accuracy_loss_interval = None
    service_test_accuracy_loss_batch_size = None

    verbose = False

    def print(self):
        s = []
        for attr in dir(self):
            if "__" in attr:
                continue
            s.append("runtime_parameters.%s = %r" % (attr, getattr(self, attr)))
        return "\n".join(s)

class Checkpoint(object):
    current_model_stat = None
    current_optimizer_stat = None
    start_model_stat = None
    end_model_stat = None
    init_model_stat = None

    current_runtime_parameter = None
    current_general_parameter = None
    current_move_parameter = None
    current_train_parameter = None
    current_rebuild_norm_parameter = None
