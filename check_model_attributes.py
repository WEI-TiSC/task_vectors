# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : check_model_attributes.py
# @Time : 2024/11/6 14:26
# Interpretation
import torch
import json

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.task_vectors import TaskVector

# # Configs
# victim_task = 'EuroSAT'
# free_rider_task = 'MNIST'
# model = 'ViT-B-32'
# scaling_coef = 0.8
#
# args = parse_arguments()
# args.data_location = 'data'
# args.model = model
# args.save = f'checkpoints/{model}'
#
# victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'  # Vector to be merged...
# pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'  # Pre-trained checkpoint for T_source
# free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'  # Theta_dest, who wants T_source
# victm_task_checkpoint_permuted = (f'permuted models/white box/{model}/{victim_task}/'
#                                   f'victim_{victim_task}_fr_{free_rider_task}_permuted.pt')
#
# victim_encoder = torch.load(victim_task_checkpoint)
# victim_permuted_encoder = torch.load(victm_task_checkpoint_permuted)
# free_rider_encoder = torch.load(free_rider_task_checkpoint)
#
#
# if __name__ == "__main__":
#     results_dict = {'scaling coef': scaling_coef}
#
#     # Benign victim performance
#     print(f"Evaluating Fine-tuned Source Performance on victim: {victim_task}...")
#     victim_info = eval_single_dataset(victim_encoder, victim_task, args)
#     results_dict[victim_task + '_benign'] = victim_info['top1']
#
#     # Bengin permuted victim performance
#     print(f"Evaluating Fine-tuned Source Performance on permuted victim: {victim_task}...")
#     victim_permuted_info = eval_single_dataset(victim_permuted_encoder, victim_task, args)
#     results_dict[victim_task + '_permuted'] = victim_permuted_info['top1']
#
#     # Generate Task Vector
#     T_victim = TaskVector(pretrained_checkpoint, victim_task_checkpoint)  # T_source = theta_source - theta_pre
#     T_victim_permuted = TaskVector(pretrained_checkpoint, victm_task_checkpoint_permuted)
#     T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)  # T_dest = theta_dest - theta_pre
#
#     # Model merging via TA
#     combine_task_T = sum([T_victim, T_free_rider])
#     combine_task_T_permuted = sum([T_victim_permuted, T_free_rider])
#     combined_encoder = combine_task_T.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
#     combined_encoder_permuted = combine_task_T_permuted.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
#
#     # Evaluate task arithmetic
#     print(f"\nEvaluating... vic:{victim_task}, fr:{free_rider_task} (Scaling coef:"
#           f" {scaling_coef}): on task: {victim_task}...")
#     ta_info = eval_single_dataset(combined_encoder, victim_task, args)
#     results_dict[victim_task + '_TA'] = ta_info['top1']
#
#     # Evaluate permuted arithmetic
#     print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef:"
#           f" {scaling_coef}): on task: {victim_task}...")
#     permuted_ta_info = eval_single_dataset(combined_encoder_permuted, victim_task, args)
#     results_dict[victim_task + '_TA_permuted'] = permuted_ta_info['top1']
#
#     # Evaluate permuted arithmetic for free-rider performance
#     print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef:"
#           f" {scaling_coef}): on task: {free_rider_task}...")
#     permuted_ta_fr_info = eval_single_dataset(combined_encoder_permuted, free_rider_task, args)
#     results_dict[free_rider_task + '_TA_permuted'] = permuted_ta_fr_info['top1']
#
#     record = f'permuted models/white box/{model}/{victim_task}/results/'\
#              f'victim_{victim_task}_fr_{free_rider_task}_sc_{scaling_coef}.txt'
#     with open(record, 'w') as f:
#         json.dump(results_dict, f, indent=4)

if __name__ == "__main__":
    # Get command line params
    args = parse_arguments()
    victim_task = args.victim_task
    free_rider_task = args.free_rider_task
    model = args.base_model
    scaling_coef = args.scaling_coef

    args.data_location = 'data'
    args.model = model
    args.save = f'checkpoints/{model}'

    victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'
    victm_task_checkpoint_permuted = (f'permuted models/white box/{model}/{victim_task}/'
                                      f'victim_{victim_task}_fr_{free_rider_task}_permuted.pt')

    # Load Model
    victim_encoder = torch.load(victim_task_checkpoint)
    victim_permuted_encoder = torch.load(victm_task_checkpoint_permuted)
    free_rider_encoder = torch.load(free_rider_task_checkpoint)

    results_dict = {'scaling coef': scaling_coef}

    # # Evaluate benign performance
    # print(f"Evaluating Fine-tuned Source Performance on victim: {victim_task}...")
    # victim_info = eval_single_dataset(victim_encoder, victim_task, args)
    # results_dict[victim_task + '_benign'] = victim_info['top1']
    #
    # # Evaluate permuted performance
    # print(f"Evaluating Fine-tuned Source Performance on permuted victim: {victim_task}...")
    # victim_permuted_info = eval_single_dataset(victim_permuted_encoder, victim_task, args)
    # results_dict[victim_task + '_permuted'] = victim_permuted_info['top1']

    # Generate task vector
    T_victim = TaskVector(pretrained_checkpoint, victim_task_checkpoint)
    T_victim_permuted = TaskVector(pretrained_checkpoint, victm_task_checkpoint_permuted)
    T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)

    # Task arithmetic
    combine_task_T = sum([T_victim, T_free_rider])
    combine_task_T_permuted = sum([T_victim_permuted, T_free_rider])
    combined_encoder = combine_task_T.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    combined_encoder_permuted = combine_task_T_permuted.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)

    # Evaluate benign task arithmetic on victim -!
    print(f"\nEvaluating... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {victim_task}...")
    ta_info = eval_single_dataset(combined_encoder, victim_task, args)
    results_dict[victim_task + '_TA'] = ta_info['top1']

    # Evaluate benign task arithmetic on free-rider -!
    print(f"\nEvaluating... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {free_rider_task}...")
    ta_fr_info = eval_single_dataset(combined_encoder, free_rider_task, args)
    results_dict[free_rider_task + '_TA'] = ta_fr_info['top1']

    # Evaluate permuted task arithmetic on victim -!
    print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {victim_task}...")
    permuted_ta_info = eval_single_dataset(combined_encoder_permuted, victim_task, args)
    results_dict[victim_task + '_TA_permuted'] = permuted_ta_info['top1']

    # Evaluate permuted task arithmetic on free-rider -!
    print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {free_rider_task}...")
    permuted_ta_fr_info = eval_single_dataset(combined_encoder_permuted, free_rider_task, args)
    results_dict[free_rider_task + '_TA_permuted'] = permuted_ta_fr_info['top1']

    # Save results
    record = f'permuted models/white box/{model}/{victim_task}/results/'\
             f'victim_{victim_task}_fr_{free_rider_task}_sc_{scaling_coef}.txt'
    with open(record, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {record}")
