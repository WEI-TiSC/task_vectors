import sys
import os
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_permutation_inverse_scatter(P_dict, Q_dict):
    """
    可视化 Q_i 是否为 P_i 的逆 permutation，使用 scatter plot 直观展示。
    """
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 8))

    for i in range(12):
        P_i = P_dict[f'P_{i}'].numpy()
        Q_i = Q_dict[f'P_{i}'].numpy()

        recovered_indices = Q_i[P_i]  # 计算恢复后的索引
        ground_truth = np.arange(3072)

        ax = axes[i // 4, i % 4]
        ax.scatter(ground_truth, recovered_indices, s=1, alpha=0.6, label="Recovered")
        ax.plot(ground_truth, ground_truth, color='r', linestyle='--', linewidth=1, label="y=x (Ideal)")
        ax.set_title(f"P'_{i} applied to P_{i}")
        ax.set_xlabel("Original Index")
        ax.set_ylabel("Recovered Index")
        ax.legend()

    plt.tight_layout()
    save_path = os.path.join('experiments/adaptive_free_rider/visual_P_and_Pprime', 'P_and_Pprime_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_permutation_heatmap(permutation_dict, title="Permutation Heatmap (P_0 to P_11)"):
    """
    可视化 permutation 矩阵的热力图。

    参数:
        permutation_dict (dict): 存储 P_0 到 P_n 的字典，每个 P_i 是一个 permutation 索引 tensor。
        title (str): 图的标题，默认 "Permutation Heatmap (P_0 to P_11)"。
    """
    # 获取所有 P_i 的 key
    keys = list(permutation_dict.keys())

    # 初始化热力图数据矩阵
    num_permutations = len(keys)
    heatmap_data = np.zeros((num_permutations, 3072))

    for i, key in enumerate(keys):
        P_i = permutation_dict[key].numpy()  # 获取 P_i 的 numpy 数组
        heatmap_data[i, P_i] = np.arange(3072)  # 用 permutation 值填充

    # 绘制热力图
    plt.figure(figsize=(16, 6))
    sns.heatmap(heatmap_data, cmap="Blues", cbar=True, xticklabels=100, yticklabels=keys)

    plt.xlabel("Original Index")
    plt.ylabel("Permutation Set")
    plt.title(title)
    plt.show()


def plot_PQ_error_heatmap(P_dict, Q_dict, save_path="PQ_error_heatmap.png"):
    """
    绘制 P 和 Q 作用后的误差热力图，检查 Q 是否完全还原 P，并保存图片。

    参数:
        P_dict (dict): 存储 P_0 到 P_11 的字典，每个 P_i 是一个 permutation 索引 tensor。
        Q_dict (dict): 存储 Q_0 到 Q_11 的字典，每个 Q_i 是一个 permutation 索引 tensor。
        save_path (str): 图片保存路径，默认 "PQ_error_heatmap.png"。
    """
    num_permutations = len(P_dict)
    error_matrix = np.zeros((num_permutations, 3072))

    for i in range(num_permutations):
        P_i = P_dict[f'P_{i}'].numpy()
        Q_i = Q_dict[f'P_{i}'].numpy()
        recovered_indices = Q_i[P_i]  # Q 作用在 P 上

        error_matrix[i] = recovered_indices - np.arange(3072)  # 计算误差

    # 画出误差热力图
    plt.figure(figsize=(16, 6))
    sns.heatmap(error_matrix, cmap="coolwarm", center=0, cbar=True, xticklabels=100,
                yticklabels=[f'P_{i}' for i in range(num_permutations)])

    plt.xlabel("Original Index")
    plt.ylabel("Permutation Set")
    plt.title("Error Heatmap of P and P' - Identity")

    # 保存图片
    save_path = os.path.join('experiments/adaptive_free_rider/visual_P_and_Pprime', save_path)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 误差热力图已保存至 {save_path}")

    plt.show()


if __name__ == "__main__":
    Ps = torch.load("permutation_dict.pt")
    P_primes = torch.load("reversed_permutation_dict.pt")

    for i in range(7, 12):
        Ps[f'P_{i}'] = torch.arange(3072, dtype=torch.int32)

    plot_permutation_inverse_scatter(Ps, P_primes)
    plot_PQ_error_heatmap(Ps, P_primes)