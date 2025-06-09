import copy
import enum
import torch

from py_src import special_torch_layers

class VarianceCorrectionType(enum.Enum):
    FollowSelfVariance = 0,
    FollowConservative = 1,
    FollowOthers = 2,

class VarianceCorrector:
    def __init__(self, variance_correction_type: VarianceCorrectionType):
        self.variance_correction_type = variance_correction_type
        self.variance_record = None
        self.model_counter = 0

    @staticmethod
    def calculate_variance_for_tensor(tensor):
        if tensor.numel() == 1:
            return 0.0
        else:
            return torch.var(tensor).item()

    def add_variance(self, model_stat):
        if (self.variance_correction_type == VarianceCorrectionType.FollowConservative) or (self.variance_correction_type == VarianceCorrectionType.FollowOthers):
            if self.variance_record is None:
                self.variance_record = {}
                for name, param in model_stat.items():
                    self.variance_record[name] = 0.0
            for name, param in model_stat.items():
                if special_torch_layers.is_ignored_layer_variance_correction(name):
                    continue
                variance = self.calculate_variance_for_tensor(param)
                self.variance_record[name] += variance
            self.model_counter += 1

    def get_variance(self, self_model_stat=None, conservative: float | None = None):

        output = None
        if self.variance_correction_type == VarianceCorrectionType.FollowSelfVariance:
            assert self_model_stat is not None
            assert conservative is not None
            conservative = float(conservative)
            self_variance = {}
            for name, param in self_model_stat.items():
                self_variance[name] = self.calculate_variance_for_tensor(param)
            output = self_variance
        if self.variance_correction_type == VarianceCorrectionType.FollowConservative:
            assert self.variance_record is not None
            assert self_model_stat is not None
            assert conservative is not None
            conservative = float(conservative)
            output = {}
            self_variance = {}
            for name, param in self_model_stat.items():
                self_variance[name] = self.calculate_variance_for_tensor(param)
            for name, var in self.variance_record.items():
                output[name] = self_variance[name] * conservative + (var/self.model_counter) * (1-conservative)
        if self.variance_correction_type == VarianceCorrectionType.FollowOthers:
            assert self.variance_record is not None
            output = {}
            for name, var in self.variance_record.items():
                output[name] = var/self.model_counter

        self.model_counter = 0
        self.variance_record = None
        return output

    @staticmethod
    def scale_tensor_to_variance(layer_tensor, target_variance):
        with torch.no_grad():
            epsilon = 0.0001
            if layer_tensor.numel() == 1:
                assert target_variance == 0.0, f"var {target_variance} != 0.0"
                rescaled_tensor = layer_tensor
            else:
                current_mean = torch.mean(layer_tensor)
                current_variance = torch.var(layer_tensor)
                scaling_factor = torch.sqrt( (target_variance + epsilon) / (current_variance + epsilon) )
                rescaled_tensor = (layer_tensor - current_mean) * scaling_factor + current_mean
        return rescaled_tensor

    @staticmethod
    def scale_model_stat_to_variance(model_stat, target_variance, ignore_layer_list=None):
        output_model_stat = copy.deepcopy(model_stat)
        for layer_name, single_layer_variance in target_variance.items():
            if special_torch_layers.is_ignored_layer_variance_correction(layer_name):
                continue
            if layer_name in [ignore_layer_list]:
                continue
            output_model_stat[layer_name] = VarianceCorrector.scale_tensor_to_variance(model_stat[layer_name], single_layer_variance)
        return output_model_stat
