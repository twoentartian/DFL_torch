import argparse
import logging
import os
import sys

import torch
from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from py_src import util
from tool.autoNEB.autoneb_utils import clone_state_dict, resolve_model_and_setup


def parse_bool_flag(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"unable to parse boolean value from {value!r}")


def build_default_output_path(model_a_path: str, model_b_path: str, match_opposite: bool) -> str:
    model_a_name = os.path.splitext(os.path.splitext(os.path.basename(model_a_path))[0])[0]
    model_b_name = os.path.splitext(os.path.splitext(os.path.basename(model_b_path))[0])[0]
    opposite_suffix = "_opposite" if match_opposite else ""
    file_name = f"{model_a_name}_permuted_to_{model_b_name}{opposite_suffix}.model.pt"
    return os.path.join(os.path.dirname(model_a_path) or os.curdir, file_name)


def pairwise_squared_distance(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    a_norm = torch.sum(features_a * features_a, dim=1, keepdim=True)
    b_norm = torch.sum(features_b * features_b, dim=1, keepdim=True).transpose(0, 1)
    distances = a_norm + b_norm - 2.0 * (features_a @ features_b.transpose(0, 1))
    return torch.clamp(distances, min=0.0)


def solve_assignment(cost_matrix: torch.Tensor) -> list[int]:
    row_ind, col_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
    assignment_by_target = [-1 for _ in range(cost_matrix.shape[1])]
    for row_index, col_index in zip(row_ind.tolist(), col_ind.tolist()):
        assignment_by_target[col_index] = row_index
    if any(index < 0 for index in assignment_by_target):
        raise RuntimeError("assignment solver returned an incomplete permutation")
    return assignment_by_target


def apply_target_sign(features: torch.Tensor, match_opposite: bool) -> torch.Tensor:
    return -features if match_opposite else features


def get_block_prefix(block_index: int) -> str:
    return f"classifier.blocks.{block_index}"


def extract_mlp_unit_features(state_dict: dict[str, torch.Tensor], block_index: int) -> torch.Tensor:
    prefix = get_block_prefix(block_index)
    linear1_weight = state_dict[f"{prefix}.linear1.weight"].detach().float()
    linear1_bias = state_dict[f"{prefix}.linear1.bias"].detach().float().unsqueeze(1)
    linear2_weight = state_dict[f"{prefix}.linear2.weight"].detach().float().transpose(0, 1)
    return torch.cat([linear1_weight, linear1_bias, linear2_weight], dim=1)


def extract_attention_head_features(
    state_dict: dict[str, torch.Tensor],
    block_index: int,
    num_heads: int,
) -> torch.Tensor:
    prefix = get_block_prefix(block_index)
    qkv_weight = state_dict[f"{prefix}.self_attn.qkv.weight"].detach().float()
    proj_weight = state_dict[f"{prefix}.self_attn.proj.weight"].detach().float()

    embed_dim = proj_weight.shape[0]
    head_dim = embed_dim // num_heads
    qkv_heads = qkv_weight.reshape(3, num_heads, head_dim, embed_dim)
    proj_input_heads = proj_weight.reshape(embed_dim, num_heads, head_dim).permute(1, 0, 2)

    features = []
    for head_index in range(num_heads):
        pieces = [
            qkv_heads[0, head_index].reshape(-1),
            qkv_heads[1, head_index].reshape(-1),
            qkv_heads[2, head_index].reshape(-1),
            proj_input_heads[head_index].reshape(-1),
        ]
        features.append(torch.cat(pieces))
    return torch.stack(features, dim=0)


def find_mlp_permutation(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    block_index: int,
    match_opposite: bool,
) -> list[int]:
    features_a = extract_mlp_unit_features(state_a, block_index)
    features_b = apply_target_sign(extract_mlp_unit_features(state_b, block_index), match_opposite)
    cost_matrix = pairwise_squared_distance(features_a, features_b)
    return solve_assignment(cost_matrix)


def find_attention_head_permutation(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    block_index: int,
    num_heads: int,
    match_opposite: bool,
) -> list[int]:
    features_a = extract_attention_head_features(state_a, block_index, num_heads)
    features_b = apply_target_sign(extract_attention_head_features(state_b, block_index, num_heads), match_opposite)
    cost_matrix = pairwise_squared_distance(features_a, features_b)
    return solve_assignment(cost_matrix)


def apply_mlp_permutation(state_dict: dict[str, torch.Tensor], block_index: int, permutation: list[int]) -> None:
    prefix = get_block_prefix(block_index)
    perm_tensor = torch.tensor(permutation, dtype=torch.long, device=state_dict[f"{prefix}.linear1.weight"].device)

    state_dict[f"{prefix}.linear1.weight"] = state_dict[f"{prefix}.linear1.weight"].index_select(0, perm_tensor).clone()
    state_dict[f"{prefix}.linear1.bias"] = state_dict[f"{prefix}.linear1.bias"].index_select(0, perm_tensor).clone()
    state_dict[f"{prefix}.linear2.weight"] = state_dict[f"{prefix}.linear2.weight"].index_select(1, perm_tensor).clone()


def apply_attention_head_permutation(
    state_dict: dict[str, torch.Tensor],
    block_index: int,
    permutation: list[int],
    num_heads: int,
) -> None:
    prefix = get_block_prefix(block_index)

    qkv_weight = state_dict[f"{prefix}.self_attn.qkv.weight"]
    proj_weight = state_dict[f"{prefix}.self_attn.proj.weight"]

    embed_dim = proj_weight.shape[0]
    head_dim = embed_dim // num_heads
    perm_tensor = torch.tensor(permutation, dtype=torch.long, device=qkv_weight.device)

    qkv_heads = qkv_weight.reshape(3, num_heads, head_dim, embed_dim)
    qkv_heads = qkv_heads.index_select(1, perm_tensor).reshape_as(qkv_weight)
    state_dict[f"{prefix}.self_attn.qkv.weight"] = qkv_heads.clone()

    proj_input_heads = proj_weight.reshape(embed_dim, num_heads, head_dim)
    proj_input_heads = proj_input_heads.index_select(1, perm_tensor).reshape_as(proj_weight)
    state_dict[f"{prefix}.self_attn.proj.weight"] = proj_input_heads.clone()


def compute_group_distance(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    difference = features_a - features_b
    return float(torch.sum(difference * difference).item())


def summarize_block_distances(
    state_a_before: dict[str, torch.Tensor],
    state_a_after: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    block_index: int,
    num_heads: int,
    match_opposite: bool,
) -> dict[str, float]:
    target_mlp = apply_target_sign(extract_mlp_unit_features(state_b, block_index), match_opposite)
    target_heads = apply_target_sign(extract_attention_head_features(state_b, block_index, num_heads), match_opposite)

    mlp_before = extract_mlp_unit_features(state_a_before, block_index)
    mlp_after = extract_mlp_unit_features(state_a_after, block_index)
    heads_before = extract_attention_head_features(state_a_before, block_index, num_heads)
    heads_after = extract_attention_head_features(state_a_after, block_index, num_heads)

    return {
        "mlp_before": compute_group_distance(mlp_before, target_mlp),
        "mlp_after": compute_group_distance(mlp_after, target_mlp),
        "heads_before": compute_group_distance(heads_before, target_heads),
        "heads_after": compute_group_distance(heads_after, target_heads),
    }


def permute_cct_7_3x1_32(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
    model: torch.nn.Module,
    match_opposite: bool,
    logger: logging.Logger,
) -> tuple[dict[str, torch.Tensor], list[dict[str, object]]]:
    output_state = clone_state_dict(state_a)
    state_before_all = clone_state_dict(state_a)

    classifier = model.classifier
    block_count = len(classifier.blocks)
    num_heads = classifier.blocks[0].self_attn.num_heads

    block_summaries: list[dict[str, object]] = []
    for block_index in range(block_count):
        head_permutation = find_attention_head_permutation(output_state, state_b, block_index, num_heads, match_opposite)
        apply_attention_head_permutation(output_state, block_index, head_permutation, num_heads)

        mlp_permutation = find_mlp_permutation(output_state, state_b, block_index, match_opposite)
        apply_mlp_permutation(output_state, block_index, mlp_permutation)

        distance_summary = summarize_block_distances(
            state_before_all,
            output_state,
            state_b,
            block_index,
            num_heads,
            match_opposite,
        )
        block_summary = {
            "block_index": block_index,
            "head_permutation": head_permutation,
            "mlp_permutation_preview": mlp_permutation[:16],
            "mlp_permutation_size": len(mlp_permutation),
            **distance_summary,
        }
        block_summaries.append(block_summary)
        logger.info(
            "block %d: head distance %.6e -> %.6e, mlp distance %.6e -> %.6e",
            block_index,
            distance_summary["heads_before"],
            distance_summary["heads_after"],
            distance_summary["mlp_before"],
            distance_summary["mlp_after"],
        )
        state_before_all = clone_state_dict(output_state)

    return output_state, block_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Permute model A through exact symmetry operations so it better matches model B in weight space."
    )
    parser.add_argument("model_a", type=str, help="path to the source .model.pt checkpoint")
    parser.add_argument("model_b", type=str, help="path to the reference .model.pt checkpoint")
    parser.add_argument(
        "match_opposite",
        type=parse_bool_flag,
        help="whether to match the negative of model B in weight space (true/false)",
    )
    parser.add_argument("-m", "--model", type=str, default=None, help="override the model name stored in the checkpoints")
    parser.add_argument("-d", "--dataset", type=str, default=None, help="override the dataset name stored in the checkpoints")
    parser.add_argument("-P", "--torch_preset_version", type=int, default=None, help="PyTorch preset version for ml_setup")
    parser.add_argument("-o", "--output", type=str, default=None, help="output .model.pt path")

    args = parser.parse_args()

    logger = logging.getLogger("permute_model")
    util.set_logging(logger, "permute")

    if not os.path.exists(args.model_a):
        raise FileNotFoundError(f"{args.model_a} does not exist")
    if not os.path.exists(args.model_b):
        raise FileNotFoundError(f"{args.model_b} does not exist")
    if os.path.isdir(args.model_a) or os.path.isdir(args.model_b):
        raise RuntimeError("permute_model.py expects checkpoint files, not folders")

    state_a, state_b, model_name, dataset_name, current_ml_setup = resolve_model_and_setup(
        args.model_a,
        args.model_b,
        args.model,
        args.dataset,
        args.torch_preset_version,
        device=torch.device("cpu"),
    )
    logger.info("resolved model = %s, dataset = %s", model_name, dataset_name)

    if model_name != "cct_7_3x1_32":
        raise NotImplementedError(
            f"Permutation matching is not implemented yet for {model_name}; only cct_7_3x1_32 is supported."
        )

    permuted_state, block_summaries = permute_cct_7_3x1_32(
        state_a,
        state_b,
        current_ml_setup.model,
        args.match_opposite,
        logger,
    )

    output_path = args.output if args.output is not None else build_default_output_path(
        args.model_a,
        args.model_b,
        args.match_opposite,
    )
    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    util.save_model_state(output_path, permuted_state, model_name, dataset_name)
    logger.info("saved permuted model to %s", output_path)

    summary_path = f"{output_path}.summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write(f"model_a={args.model_a}\n")
        file.write(f"model_b={args.model_b}\n")
        file.write(f"match_opposite={args.match_opposite}\n")
        file.write(f"model_name={model_name}\n")
        file.write(f"dataset_name={dataset_name}\n")
        for block_summary in block_summaries:
            file.write(f"{block_summary}\n")
    logger.info("saved permutation summary to %s", summary_path)
