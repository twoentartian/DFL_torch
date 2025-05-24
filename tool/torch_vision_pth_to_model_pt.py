import argparse
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from py_src import ml_setup, util

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='torch_vision_pth_to_model_pt', description='Convert torch vision .pth file to .model.pt file')
    parser.add_argument('pth_path', type=str, help='path to .pth file')
    parser.add_argument('model_type', type=str, help='model type in .model.pt format')
    parser.add_argument('-d', '--dataset_type', type=str, default='default', help='the dataset type can used to train the model')
    parser.add_argument("--ema", help='extract the ema model')


    args = parser.parse_args()

    cpu = torch.device('cpu')
    pth_target = torch.load(args.pth_path, map_location=cpu, weights_only=False)
    if args.ema:
        model_stat_dict = pth_target['model_ema']
        output_path = f"{args.pth_path}_ema.model.pt"
    else:
        model_stat_dict = pth_target['model']
        output_path = f"{args.pth_path}.model.pt"
    current_ml_setup = ml_setup.get_ml_setup_from_config(args.model_type, args.dataset_type)
    model = current_ml_setup.model
    model.load_state_dict(model_stat_dict)
    util.save_model_state(output_path, model_stat_dict, model_name=current_ml_setup.model_name)

