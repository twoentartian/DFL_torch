from torchvision import models
from transformers import AutoTokenizer

from py_src.ml_setup_base.base import MlSetup
from py_src.ml_setup_base.model import ModelType
import py_src.ml_setup_base.dataset as ml_setup_dataset
from py_src.ml_setup_base.other_setup import get_pytorch_training_imagenet

import py_src.models.nanoclip as nanoclip
from py_src.ml_setup_base.dataset_flickr import CollateFlickr

def nanoclip_flickr30k_default():
    output_ml_setup = MlSetup()
    txt_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    dataset = ml_setup_dataset.dataset_flickr30k(txt_model=txt_model_name, img_model='dinov2_vits14')

    output_ml_setup.model = nanoclip.NanoCLIP()
    output_ml_setup.model_name = str(ModelType.nanoclip_default.name)
    output_ml_setup.model_type = ModelType.nanoclip_default
    output_ml_setup.get_info_from_dataset(dataset)
    output_ml_setup.training_batch_size = 128
    output_ml_setup.has_normalization_layer = True

    output_ml_setup.criterion = nanoclip.ContrastiveLoss(temperature=0.05)
    tokenizer = AutoTokenizer.from_pretrained(txt_model_name)
    output_ml_setup.collate_fn = CollateFlickr(tokenizer, max_length=80, captions_to_use='all')
    output_ml_setup.collate_fn_val = CollateFlickr(tokenizer,  max_length=80, captions_to_use='first')
    return output_ml_setup
