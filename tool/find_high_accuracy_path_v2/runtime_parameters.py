from enum import Enum, auto

class WorkMode(Enum):
    unknown = auto()
    to_origin = auto()
    to_inf = auto()
    to_certain_model = auto()

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

    """real-time values"""
    current_tick = None
    max_tick = None

    debug_check_config_mode = None
    test_dataset_use_whole = None

    def print(self):
        s = []
        for attr in dir(self):
            if "__" in attr:
                continue
            s.append("runtime_parameters.%s = %r" % (attr, getattr(self, attr)))
        return "\n".join(s)