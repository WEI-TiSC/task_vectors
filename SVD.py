# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : SVD.py
# @Time : 2024/9/5 23:11
# Interpretation
import torch
import torch.nn.functional as F

from src.args import parse_arguments
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset

# Configs
victim_task = 'GTSRB'
free_rider_task = 'SVHN'
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


def svd_transmission(layer: int):
    # Get Resblock
    resblock = victim_encoder.model.visual.transformer.resblocks[layer]

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


if __name__ == "__main__":
    # Benign source performance
    print(f"Evaluating Fine-tuned Source Performance on victim: {victim_task}...")
    victim_info = eval_single_dataset(victim_encoder, victim_task, args)

    # Bengin dest performance
    print(f"\nEvaluating Fine-tuned Source Performance on free-rider: {free_rider_task}...")
    free_rider_info = eval_single_dataset(free_rider_encoder, free_rider_task, args)

    T_victim = TaskVector(pretrained_checkpoint, victim_task_checkpoint)  # T_source = theta_source - theta_pre
    T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)  # T_dest = theta_dest - theta_pre
    combine_task_T = sum([T_victim, T_free_rider])
    print(f"\nEvaluating Combining vector of victim: {victim_task} and free-rider: {free_rider_task} (Scaling coef:"
          f" {vector_scaling_coef}): Performace on victim's task: {victim_task}...")
    combined_encoder = combine_task_T.apply_to(pretrained_checkpoint, scaling_coef=vector_scaling_coef)
    editted_dest_info_victim = eval_single_dataset(combined_encoder, victim_task, args)
    print(f"\nEvaluating Combining vector of victim: {victim_task} and free-rider: {free_rider_task} (Scaling coef:"
          f" {vector_scaling_coef}): Performace on free-rider's task: {free_rider_task}...")
    editted_dest_info_free_rider = eval_single_dataset(combined_encoder, free_rider_task, args)

    # SVD Setting Up - Only for source encoder
    # Orthogonal trans layers
    # source_encoder.to('cpu')
    layers = [11]

    # print(f"SVD on victim model: {victim_task}, in Resblocks: {layers}")
    # for layer in layers:
    #     svd_transmission(layer)
    #
    # torch.save(victim_encoder, f'victim_svd_{victim_task}.pt')
    # victim_svd_checkpoint = f'victim_svd_{victim_task}.pt'
    #
    # print(f"\nEvaluating SVDed victim vector on victim: {victim_task}...")
    # victim_svd_info = eval_single_dataset(victim_encoder, victim_task, args)
    #
    # svd_vic_T = TaskVector(pretrained_checkpoint, victim_svd_checkpoint)
    # combined_svd_vic_T = sum([svd_vic_T, T_free_rider])
    # print(f"\nEvaluating Combining vector of SVDed victim: {victim_task} and free-rider: {free_rider_task} (Scaling coef:"
    #       f" {vector_scaling_coef}): Performace on victim's task: {victim_task}...")
    # combined_svd_encoder = combined_svd_vic_T.apply_to(pretrained_checkpoint, scaling_coef=vector_scaling_coef)
    # svd_editted_dest_info = eval_single_dataset(combined_svd_encoder, victim_task, args)

