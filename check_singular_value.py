# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : check_singular_value.py
# @Time : 2024/10/9 22:29
# Interpretation
import os
import torch
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.args import parse_arguments
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset

# Configs
victim_task = 'DTD'
free_rider_task = 'MNIST'
model = 'ViT-B-32'
vector_scaling_coef = 0.4

args = parse_arguments()
args.data_location = 'data'
args.model = model
args.save = f'checkpoints/{model}'

victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'  # Vector to be merged...
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'  # Pre-trained checkpoint for T_source
free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'  # Theta_dest, who wants T_source

victim_encoder = torch.load(victim_task_checkpoint)
free_rider_encoder = torch.load(free_rider_task_checkpoint)


def encoder_svd(encoder, layer: int):
    # Get Resblock
    resblock = encoder.model.visual.transformer.resblocks[layer]

    # Extract FC layers
    W_fc = resblock.mlp.c_fc.weight.data.clone()
    W_proj = resblock.mlp.c_proj.weight.data.clone()

    U1, S1, V1 = torch.linalg.svd(W_fc)
    U2, S2, V2 = torch.linalg.svd(W_proj)

    W_fc_prime = V2 @ U1.T @ W_fc
    W_proj_prime = W_proj @ V2.T @ U1

    # Change weights
    resblock.mlp.c_fc.weight.data = W_fc_prime
    resblock.mlp.c_proj.weight.data = W_proj_prime

    return encoder


if __name__ == "__main__":
    modify_layer = 10
    victim_encoder = encoder_svd(victim_encoder, modify_layer)
    torch.save(victim_encoder, f'victim_svd_{victim_task}.pt')
    victim_svd_checkpoint = f'victim_svd_{victim_task}.pt'

    T_victim_svd = TaskVector(pretrained_checkpoint, victim_svd_checkpoint)  # T_source = theta_source - theta_pre
    T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)  # T_dest = theta_dest - theta_pr

    T_comb_task = sum([T_victim_svd, T_free_rider])
    merged_encoder = T_comb_task.apply_to(pretrained_checkpoint, scaling_coef=vector_scaling_coef)

    for i in [modify_layer]:
        W_vic = victim_encoder.model.visual.transformer.resblocks[i].mlp.c_proj.weight.data.clone()
        W_fr = free_rider_encoder.model.visual.transformer.resblocks[i].mlp.c_proj.weight.data.clone()
        W_m = merged_encoder.model.visual.transformer.resblocks[i].mlp.c_proj.weight.data.clone()

        _, singular_values_vic, _ = np.linalg.svd(W_vic, full_matrices=False)
        _, singular_values_fr, _ = np.linalg.svd(W_fr, full_matrices=False)
        _, singular_values_m, _ = np.linalg.svd(W_m, full_matrices=False)

        # 绘制三张子图，分别显示 Model A, B 和 M 的奇异值分布
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 绘制 Model A 的奇异值分布
        axes[0].plot(singular_values_vic, marker='o')
        axes[0].set_title('Singular Value Distribution - Model SVD Victim')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Singular Value')
        axes[0].grid(True)

        # 绘制 Model B 的奇异值分布
        axes[1].plot(singular_values_fr, marker='x')
        axes[1].set_title('Singular Value Distribution - Model free-rider')
        axes[1].set_xlabel('Index')
        axes[1].grid(True)

        # 绘制 Model M 的奇异值分布
        axes[2].plot(singular_values_m, marker='s')
        axes[2].set_title('Singular Value Distribution - Model SVD merged')
        axes[2].set_xlabel('Index')
        axes[2].grid(True)

        # 显示图像
        plt.tight_layout()
        plt.savefig(
            os.path.join(os.getcwd(), 'figs', f'Vic_{victim_task} fr_{free_rider_task} Singular Distribution of layer {i}.png'))
        plt.show()

    # T_victim = TaskVector(pretrained_checkpoint, victim_task_checkpoint)  # T_source = theta_source - theta_pre
    # T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)  # T_dest = theta_dest - theta_pre
    #
    # T_comb_task = sum([T_victim, T_free_rider])
    # merged_encoder = T_comb_task.apply_to(pretrained_checkpoint, scaling_coef=vector_scaling_coef)
    #
    # check_layers = [i for i in range(12)]
    #
    # fig, axes = plt.subplots(6, 6, figsize=(24, 24))
    #
    # for i in check_layers:
    #     W_vic = victim_encoder.model.visual.transformer.resblocks[i].mlp.c_fc.weight.data.clone()
    #     W_fr = free_rider_encoder.model.visual.transformer.resblocks[i].mlp.c_fc.weight.data.clone()
    #     W_m = merged_encoder.model.visual.transformer.resblocks[i].mlp.c_fc.weight.data.clone()
    #
    #     _, singular_values_vic, _ = np.linalg.svd(W_vic, full_matrices=False)
    #     _, singular_values_fr, _ = np.linalg.svd(W_fr, full_matrices=False)
    #     _, singular_values_m, _ = np.linalg.svd(W_m, full_matrices=False)
    #
    #     axes[i // 2, (i % 2) * 3 + 0].plot(singular_values_vic, marker='o')
    #     axes[i // 2, (i % 2) * 3 + 0].set_title(f'Model victim - Layer {i + 1}')
    #     axes[i // 2, (i % 2) * 3 + 0].set_xlabel('Index')
    #     axes[i // 2, (i % 2) * 3 + 0].set_ylabel('Singular Value')
    #     axes[i // 2, (i % 2) * 3 + 0].grid(True)
    #
    #     axes[i // 2, (i % 2) * 3 + 1].plot(singular_values_fr, marker='x')
    #     axes[i // 2, (i % 2) * 3 + 1].set_title(f'Model free-rider - Layer {i + 1}')
    #     axes[i // 2, (i % 2) * 3 + 1].set_xlabel('Index')
    #     axes[i // 2, (i % 2) * 3 + 1].set_ylabel('Singular Value')
    #     axes[i // 2, (i % 2) * 3 + 1].grid(True)
    #
    #     axes[i // 2, (i % 2) * 3 + 2].plot(singular_values_m, marker='s')
    #     axes[i // 2, (i % 2) * 3 + 2].set_title(f'Model merged - Layer {i + 1}')
    #     axes[i // 2, (i % 2) * 3 + 2].set_xlabel('Index')
    #     axes[i // 2, (i % 2) * 3 + 2].set_ylabel('Singular Value')
    #     axes[i // 2, (i % 2) * 3 + 2].grid(True)
    #
    # plt.tight_layout()
    # # plt.savefig(os.path.join(os.getcwd(), 'figs', f'Vic_{victim_task} fr_{free_rider_task} Singular Distribution.png'))
    # plt.show()

        # # 计算 W_A 和 W_M 之间的差异
        # weight_diff_A_M = np.abs(W_m - W_vic)
        # weight_diff_B_M = np.abs(W_m - W_fr)
        #
        # # 使用 seaborn 热图进行可视化
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(weight_diff_A_M, cmap='coolwarm', cbar=True)
        # plt.title('Weight Difference between Model A and Model M')
        # plt.show()
        #
        # plt.figure(figsize=(10, 8))
        # sns.heatmap(weight_diff_B_M, cmap='coolwarm', cbar=True)
        # plt.title('Weight Difference between Model B and Model M')
        # plt.show()