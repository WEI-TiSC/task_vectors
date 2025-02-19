import sys
import os
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt


def plot_attention_param_distributions(
        original_params: dict,
        scaled_params: dict,
        layer_idx: int,
        save_path: str = "./attention_scaling_plots"
):
    """
    对比单个 resblock (第 layer_idx 层) 的 Attention 参数，在 scaling 之前和之后的数值分布。
    采用折线图 (按值索引排序后绘制)，直方图 (值分布统计)，以及箱线图 (查看范围和异常值)。
    **新增图片保存功能**，默认保存在 `./attention_scaling_plots/`

    参数:
    ----
    original_params: dict
        scaling **前** 的参数字典
    scaled_params: dict
        scaling **后** 的参数字典
    layer_idx: int
        要对比的 transformer block 层索引
    save_path: str
        生成的图片保存路径（默认 `./attention_scaling_plots`）

    主要绘制:
    1. 折线图 - 展示值的形状变化
    2. 直方图 - 展示分布变化
    3. 箱线图 - 展示整体数据范围
    """

    # 创建保存目录
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_file = os.path.join(save_path, f"attention_scaling_layer_{layer_idx}.png")
    else:
        save_file = None

    # 获取该层 Attention 相关参数
    keys = {
        "Q_weight": f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight",
        "Q_bias": f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_bias",
        "K_weight": f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight",
        "K_bias": f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_bias",
        "V_weight": f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_weight",
        "V_bias": f"model.visual.transformer.resblocks.{layer_idx}.attn.in_proj_bias",
        "W_O": f"model.visual.transformer.resblocks.{layer_idx}.attn.out_proj.weight"
    }

    fig, axes = plt.subplots(2, len(keys), figsize=(18, 9))
    plt.suptitle(f"Attention Scaling Before vs. After (Layer {layer_idx})", fontsize=16)

    for i, (name, key) in enumerate(keys.items()):
        if key not in original_params or key not in scaled_params:
            continue  # 跳过不存在的键

        # 获取数据并展平为 1D
        orig = original_params[key].detach().cpu().numpy().flatten()
        scaled = scaled_params[key].detach().cpu().numpy().flatten()

        # 1. 折线图: 按索引排序后绘制
        sorted_orig = np.sort(orig)
        sorted_scaled = np.sort(scaled)
        axes[0, i].plot(sorted_orig, label="Original", color='b', alpha=0.7)
        axes[0, i].plot(sorted_scaled, label="Scaled", color='r', linestyle="dashed", alpha=0.7)
        axes[0, i].set_title(name)
        axes[0, i].legend()

        # 2. 直方图: 统计分布情况
        axes[1, i].hist(orig, bins=50, alpha=0.5, label="Original", color='b', density=True)
        axes[1, i].hist(scaled, bins=50, alpha=0.5, label="Scaled", color='r', density=True)
        axes[1, i].legend()

        # # 3. 箱线图: 查看范围与极端值
        # axes[2, i].boxplot([orig, scaled], vert=False, patch_artist=True,
        #                    labels=["Original", "Scaled"], widths=0.6)

    plt.tight_layout()

    # 保存图片
    if save_file is not None:
        plt.savefig(save_file, dpi=300, bbox_inches='tight')
        print(f"Saved attention visualization to {save_file}")

    # 显示图像
    plt.show()


if __name__ == "__main__":
    # Config for addition
    task = 'GTSRB'
    model = 'ViT-B-32'
    num_layers = 12

    task_checkpoint = f'checkpoints/{model}/{task}/finetuned.pt'
    scaling_task_checkpoint = (f'experiments/perm_all_layers/permuted models/perm_scale/'
                               f'{model}/{task}/victim_{task}_perm_scale_attn_qkvw.pt')

    image_encoder = torch.load(task_checkpoint)
    scaling_image_encoder = torch.load(scaling_task_checkpoint)
    original_params = {name: param.clone() for name, param in image_encoder.state_dict().items()}
    scaled_params = {name: param.clone() for name, param in scaling_image_encoder.state_dict().items()}

    save_path = f'experiments/compare_attn_scaling/{task}/attention_scaling_plots/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 绘制所有层的 attention 参数分布
    for idx in range(num_layers):
        plot_attention_param_distributions(original_params, scaled_params, layer_idx=idx, save_path=save_path)

