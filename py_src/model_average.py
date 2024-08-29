import copy
import torch
from py_src import special_torch_layers
from py_src.model_variance_correct import VarianceCorrectionType, VarianceCorrector

def move_tensor_toward(src_tensor, dest_tensor, step, adoptive_step):
    diff_tensor = dest_tensor - src_tensor
    norm = torch.norm(diff_tensor)
    step_from_adoptive_part = norm * adoptive_step
    real_step = step if step > step_from_adoptive_part else step_from_adoptive_part
    angle_tensor = diff_tensor / norm
    move_tensor = angle_tensor * real_step
    return src_tensor + move_tensor

def move_model_state_toward(src_model_stat, dest_model_stat, step, adoptive_step, enable_merge_bias_with_weight=False, ignore_layer_keywords=None):
    if ignore_layer_keywords is None:
        ignore_layer_keywords = []
    output_stat = copy.deepcopy(src_model_stat)
    layers_already_process = set()
    for layer_name in src_model_stat.keys():
        if layer_name in layers_already_process:
            continue
        if special_torch_layers.is_ignored_layer_averaging(layer_name):
            continue
        if special_torch_layers.is_keyword_in_layer_name(layer_name, ignore_layer_keywords):
            continue

        current_layer_processed = False
        # process associated bias tensor with weight tensor
        if enable_merge_bias_with_weight and ('weight' in layer_name):
            bias_layer_name = layer_name.replace('weight', 'bias')
            if bias_layer_name in src_model_stat.keys():
                # process bias layer and weight layer
                layers_already_process.add(bias_layer_name)
                layers_already_process.add(layer_name)
                src_model_weight: torch.Tensor = src_model_stat[layer_name]
                src_model_bias: torch.Tensor = src_model_stat[bias_layer_name]
                src_tensor = torch.cat((src_model_weight.flatten(), src_model_bias.flatten()), dim=0)
                dst_tensor = torch.cat((dest_model_stat[layer_name].flatten(), dest_model_stat[bias_layer_name].flatten()), dim=0)
                output_tensor = move_tensor_toward(src_tensor, dst_tensor, step, adoptive_step)
                output_weight_tensor, output_bias_tensor = torch.split(output_tensor, [src_model_weight.nelement(), src_model_bias.nelement()], dim=0)
                output_weight_tensor = output_weight_tensor.reshape(src_model_weight.shape)
                output_bias_tensor = output_bias_tensor.reshape(src_model_bias.shape)
                output_stat[layer_name] = output_weight_tensor
                output_stat[bias_layer_name] = output_bias_tensor
                current_layer_processed = True

        if not current_layer_processed:
            layers_already_process.add(layer_name)
            src_tensor = src_model_stat[layer_name]
            dst_tensor = dest_model_stat[layer_name]
            output_stat[layer_name] = move_tensor_toward(src_tensor, dst_tensor, step, adoptive_step)
    return output_stat

class ModelAverager:
    def __init__(self, variance_corrector=None, *args, **kwargs):
        self.variance_corrector = variance_corrector

    def add_model(self, model_stat):
        raise NotImplementedError

    def get_model(self, *args, **kwargs):
        raise NotImplementedError

    def get_model_count(self):
        raise NotImplementedError

    @staticmethod
    def _iadd_two_model(src, addition, weight_src: float = 1.0, weight_addition: float = 1.0, check_same_keys=True):
        with torch.no_grad():
            assert (not check_same_keys) or (set(src.keys()) == set(addition.keys()))
            for layer_name in src.keys():
                if weight_src == 1.0 and weight_addition == 1.0:
                    src[layer_name] += addition[layer_name]
                else:
                    src[layer_name] = src[layer_name] * weight_src + addition[layer_name] * weight_addition
            return src


class StandardModelAverager(ModelAverager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variance_corrector is not None:
            vc_type = self.variance_corrector.variance_correction_type
            assert (vc_type == VarianceCorrectionType.FollowOthers), "standard model averager only average models from others, thus only VarianceCorrectionType.FollowOthers is supported"
        self.model_buffer = None
        self.model_counter = 0

    def add_model(self, model_stat):
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                # remove ignored layers
                for layer_name in list(self.model_buffer.keys()):
                    if special_torch_layers.is_ignored_layer_averaging(layer_name):
                        del self.model_buffer[layer_name]
            else:
                self.model_buffer = ModelAverager._iadd_two_model(self.model_buffer, model_stat, check_same_keys=False)
            self.model_counter += 1
            # variance correction
            if self.variance_corrector is not None:
                self.variance_corrector.add_variance(model_stat)

    def get_model(self, self_model, *args, **kwargs):
        with torch.no_grad():
            output = copy.deepcopy(self_model)
            for layer_name, layer_weights in output.items():
                if layer_name not in self.model_buffer.keys():
                    continue
                output[layer_name] = self.model_buffer[layer_name] / self.model_counter

            # variance correction
            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance()
                for layer_name, single_layer_variance in target_variance.items():
                    if special_torch_layers.is_ignored_layer_variance_correction(layer_name):
                        continue
                    output[layer_name] = VarianceCorrector.scale_tensor_to_variance(output[layer_name], single_layer_variance)
            self.model_buffer = None
            self.model_counter = 0
            return output

    def get_model_count(self):
        return self.model_counter


class ConservativeModelAverager(ModelAverager):
    def __init__(self, conservative: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert (conservative >= 0) and (conservative <= 1)
        self.model_buffer = None
        self.model_counter = 0
        self.conservative = conservative

    def add_model(self, model_stat):
        with torch.no_grad():
            if self.model_buffer is None:
                self.model_buffer = copy.deepcopy(model_stat)
                # remove ignored layers
                for layer_name in list(self.model_buffer.keys()):
                    if special_torch_layers.is_ignored_layer_averaging(layer_name):
                        del self.model_buffer[layer_name]
            else:
                self.model_buffer = ModelAverager._iadd_two_model(self.model_buffer, model_stat, check_same_keys=False)
            self.model_counter += 1
            # variance correction
            if self.variance_corrector is not None:
                self.variance_corrector.add_variance(model_stat)

    def get_model(self, self_model, *args, **kwargs):
        with torch.no_grad():
            output = copy.deepcopy(self_model)
            for layer_name, layer_weights in output.items():
                if layer_name not in self.model_buffer.keys():
                    continue
                output[layer_name] = self.model_buffer[layer_name] / self.model_counter
            output = ModelAverager._iadd_two_model(self_model, output, weight_src=self.conservative, weight_addition=1 - self.conservative)
            # variance correction
            if self.variance_corrector is not None:
                target_variance = self.variance_corrector.get_variance(self_model, self.conservative)
                for layer_name, single_layer_variance in target_variance:
                    if special_torch_layers.is_ignored_layer_variance_correction(layer_name):
                        continue
                    output[layer_name] = VarianceCorrector.scale_tensor_to_variance(output[layer_name], single_layer_variance)
            self.model_buffer = None
            self.model_counter = 0
            return output

    def get_model_count(self):
        return self.model_counter

