import json
import sys
import os
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

import torch


def get_tensor_stats(tensor: torch.Tensor):
    """
    返回一个字典，包含给定张量的 max、min、abs_ratio (max绝对值 / min绝对值)。
    如果需要防止分母为 0，则可修改 eps=1e-12
    """
    tensor = tensor.abs()  # 统计绝对值！
    max_val = tensor.max().item()
    min_val = tensor.min().item()

    # 计算绝对值比
    abs_max = abs(max_val)
    abs_min = abs(min_val)
    # eps = 1e-12
    eps = 0
    ratio = (abs_max + eps) / (abs_min + eps)

    return {
        "max": max_val,
        "min": min_val,
        "ratio": ratio
    }


def gather_resblock_param_stats(params: dict, num_layers: int):
    """
    遍历 ViT 的每个 resblock (0 ~ num_layers-1)，
    收集  Attention (in_proj/out_proj) 中
    常见权重、偏置的 max/min/ratio 信息，返回字典。

    参数:
      - MLP:
        "model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight"
        "model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.bias"
        "model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight"
        "model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.bias"
      - Attention:
        "model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight"
        "model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_bias"
        "model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.weight"
        "model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.bias"
    """
    stats = {}
    global_max_ratio = -float("inf")  # 初始化全局最大比例
    global_min_ratio = float("inf")   # 初始化全局最小比例

    for layer_idx in range(num_layers):
        layer_stats = {}

        # ---------- MLP c_fc ----------
        # key_c_fc_weight = f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight"
        # key_c_fc_bias = f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.bias"
        # if key_c_fc_weight in params:
        #     layer_stats["mlp_c_fc_weight"] = get_tensor_stats(params[key_c_fc_weight])
        # if key_c_fc_bias in params:
        #     layer_stats["mlp_c_fc_bias"] = get_tensor_stats(params[key_c_fc_bias])

        # ---------- MLP c_proj ----------
        # key_c_proj_weight = f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight"
        # key_c_proj_bias = f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.bias"
        # if key_c_proj_weight in params:
        #     layer_stats["mlp_c_proj_weight"] = get_tensor_stats(params[key_c_proj_weight])
        # if key_c_proj_bias in params:
        #     layer_stats["mlp_c_proj_bias"] = get_tensor_stats(params[key_c_proj_bias])

        # ---------- Attention in_proj ----------
        key_in_proj_weight = f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight"
        key_in_proj_bias = f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_bias"
        if key_in_proj_weight in params:
            layer_stats["attn_in_proj_weight"] = get_tensor_stats(params[key_in_proj_weight])
        if key_in_proj_bias in params:
            layer_stats["attn_in_proj_bias"] = get_tensor_stats(params[key_in_proj_bias])

        # ---------- Attention out_proj ----------
        key_out_proj_weight = f"model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.weight"
        key_out_proj_bias = f"model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.bias"
        if key_out_proj_weight in params:
            layer_stats["attn_out_proj_weight"] = get_tensor_stats(params[key_out_proj_weight])
        if key_out_proj_bias in params:
            layer_stats["attn_out_proj_bias"] = get_tensor_stats(params[key_out_proj_bias])

        stats[f"resblock_{layer_idx}"] = layer_stats

        # 计算该层的最大/最小 ratio
        layer_ratios = [layer_stats[key]["ratio"] for key in layer_stats.keys()]
        if layer_ratios:
            layer_max_ratio = max(layer_ratios)
            layer_min_ratio = min(layer_ratios)

            global_max_ratio = max(global_max_ratio, layer_max_ratio)
            global_min_ratio = min(global_min_ratio, layer_min_ratio)

        stats[f"resblock_{layer_idx}"] = layer_stats

    # 存储全局最大/最小 ratio
    stats["global_max_ratio"] = global_max_ratio
    stats["global_min_ratio"] = global_min_ratio

    return stats


if __name__ == "__main__":
    # Config for addition
    task = 'GTSRB'
    model = 'ViT-B-32'
    vector_scaling_coef = 0.4
    num_layers = 12

    task_checkpoint = f'checkpoints/{model}/{task}/finetuned.pt'
    scaling_task_checkpoint = (f'experiments/perm_all_layers/permuted models/perm_scale/'
                               f'{model}/{task}/victim_{task}_perm_scale_attn_qkvw.pt')

    image_encoder = torch.load(task_checkpoint)
    scaling_image_encoder = torch.load(scaling_task_checkpoint)
    ori_params = {name: param.clone() for name, param in image_encoder.state_dict().items()}
    scaling_params = {name: param.clone() for name, param in scaling_image_encoder.state_dict().items()}

    ori_stats_dict = gather_resblock_param_stats(ori_params, num_layers)
    scaling_stats_dict = gather_resblock_param_stats(scaling_params, num_layers)

    record_dir = os.path.join(os.path.dirname(__file__), f'{task}')
    os.makedirs(record_dir, exist_ok=True)
    record_name = f'param_stastic.txt'
    scaling_record_name = f'scaling_param_stastic.txt'
    ori_record = os.path.join(record_dir, record_name)
    scaling_record = os.path.join(record_dir, scaling_record_name)

    with open(ori_record, 'w') as f:
        json.dump(ori_stats_dict, f, indent=4)
    with open(scaling_record, 'w') as f:
        json.dump(scaling_stats_dict, f, indent=4)
