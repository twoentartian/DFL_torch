from py_src.simulation_runtime_parameters import RuntimeParameters

class Service(object):
    @staticmethod
    def get_service_name() -> str:
        raise NotImplementedError

    def initialize(self, parameters: RuntimeParameters, output_path, *args, **kwargs):
        raise NotImplementedError

    def trigger(self, parameters: RuntimeParameters, *args, **kwargs):
        raise NotImplementedError

    def continue_from_checkpoint(self, checkpoint_folder_path: str, restore_until_tick: int, *args, **kwargs):
        raise NotImplementedError