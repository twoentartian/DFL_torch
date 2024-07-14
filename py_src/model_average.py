import copy
from py_src.model_variance_correct import VarianceCorrectionType, VarianceCorrector

class ModelAverager():
    def __init__(self, variance_corrector=None, *args, **kwargs):
        self.variance_corrector = variance_corrector

    def add_model(self, model_stat):
        raise NotImplementedError

    def get_model(self, *args, **kwargs):
        raise NotImplementedError

    def get_model_count(self):
        raise NotImplementedError

    @staticmethod
    def _iadd_two_model(src, addition, weight_src: float = 1.0, weight_addition: float = 1.0):
        assert set(src.keys()) == set(addition.keys())
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
        if self.model_buffer is None:
            self.model_buffer = copy.deepcopy(model_stat)
        else:
            self.model_buffer = ModelAverager._iadd_two_model(self.model_buffer, model_stat)
        self.model_counter += 1
        # variance correction
        if self.variance_corrector is not None:
            self.variance_corrector.add_variance(model_stat)

    def get_model(self, *args, **kwargs):
        for layer_name, layer_weights in self.model_buffer.items():
            if "num_batches_tracked" in layer_name:  # skip "num_batches_tracked"
                continue
            layer_weights /= self.model_counter
        output = self.model_buffer
        # variance correction
        if self.variance_corrector is not None:
            target_variance = self.variance_corrector.get_variance()
            for layer_name, single_layer_variance in target_variance.items():
                output[layer_name] = VarianceCorrector.scale_model_stat_to_variance(output[layer_name], single_layer_variance)
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
        if self.model_buffer is None:
            self.model_buffer = copy.deepcopy(model_stat)
        else:
            self.model_buffer = ModelAverager._iadd_two_model(self.model_buffer, model_stat)
        self.model_counter += 1
        # variance correction
        if self.variance_corrector is not None:
            self.variance_corrector.add_variance(model_stat)


    def get_model(self, self_model, *args, **kwargs):
        for layer_name, layer_weights in self.model_buffer.items():
            layer_weights /= self.model_counter
        output = self.model_buffer
        output = ModelAverager._iadd_two_model(output, self_model, weight_src=self.conservative, weight_addition=1 - self.conservative)
        # variance correction
        if self.variance_corrector is not None:
            target_variance = self.variance_corrector.get_variance(self_model, self.conservative)
            for layer_name, single_layer_variance in target_variance:
                output[layer_name] = VarianceCorrector.scale_model_stat_to_variance(output[layer_name], single_layer_variance)
        self.model_buffer = None
        self.model_counter = 0
        return output

    def get_model_count(self):
        return self.model_counter
