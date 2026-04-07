"""
Extract current_model_stat from a checkpoint file and save it as a .model.pt file
in the same format as util.save_model_state.

Usage:
    python tool/extract_model_from_checkpoint.py <checkpoint_file> [output_model_pt]

If output_model_pt is not specified, it is derived from the checkpoint filename:
    e.g. checkpoint_1000.checkpoint.pt -> checkpoint_1000.model.pt
"""

import sys
import os
import torch


def extract_model_from_checkpoint(checkpoint_path: str, output_path: str):
    cpu_device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=cpu_device, weights_only=False)

    model_stat = checkpoint.current_model_stat
    runtime_params = checkpoint.current_runtime_parameter
    model_name = runtime_params.model_name if runtime_params is not None else None
    dataset_name = runtime_params.dataset_name if runtime_params is not None else None

    info = {}
    info["state_dict"] = model_stat
    info["model_name"] = model_name
    info["dataset_name"] = dataset_name
    torch.save(info, output_path)

    print(f"Saved model state to: {output_path}")
    print(f"  model_name:   {model_name}")
    print(f"  dataset_name: {dataset_name}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    checkpoint_path = sys.argv[1]

    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        basename = os.path.basename(checkpoint_path)
        # e.g. checkpoint_1000.checkpoint.pt -> checkpoint_1000.model.pt
        output_basename = basename.replace(".checkpoint.pt", ".model.pt")
        if output_basename == basename:
            output_basename = basename.rsplit(".", 1)[0] + ".model.pt"
        output_path = os.path.join(os.path.dirname(checkpoint_path), output_basename)

    extract_model_from_checkpoint(checkpoint_path, output_path)
