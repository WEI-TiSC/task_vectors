import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import json

from src.args import parse_arguments
from src.eval import eval_single_dataset
from src.task_vectors import TaskVector


if __name__ == "__main__":
    # Get command line params
    args = parse_arguments()
    victim_task = args.victim_task
    free_rider_task = args.free_rider_task
    model = args.base_model
    scaling_coef = args.scaling_coef
    layers = eval(args.perm_layers)
    perm_layer_num = len(layers)
    permed_layers = '_'.join(str(layer) for layer in layers)

    args.data_location = 'data'
    args.model = model
    args.save = f'checkpoints/{model}'

    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'
    victm_task_checkpoint_permuted = (f'experiments/partial_permutation/blackbox_perm_models/{model}/{victim_task}/'
                                      f'{perm_layer_num}_layers/{permed_layers}/partial_permuted.pt')

    # Load Model
    victim_permuted_encoder = torch.load(victm_task_checkpoint_permuted)
    free_rider_encoder = torch.load(free_rider_task_checkpoint)

    results_dict = {'scaling coef': scaling_coef}

    # Generate task vector
    T_victim_permuted = TaskVector(pretrained_checkpoint, victm_task_checkpoint_permuted)
    T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)

    # Task arithmetic
    combine_task_T_permuted = sum([T_victim_permuted, T_free_rider])
    combined_encoder_permuted = combine_task_T_permuted.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)

    # Evaluate permuted task arithmetic on victim -!
    print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {victim_task}...")
    permuted_ta_info = eval_single_dataset(combined_encoder_permuted, victim_task, args)
    results_dict[victim_task + '_TA_permuted'] = permuted_ta_info['top1']

    # Evaluate permuted task arithmetic on free-rider -!
    print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {free_rider_task}...")
    permuted_ta_fr_info = eval_single_dataset(combined_encoder_permuted, free_rider_task, args)
    results_dict[free_rider_task + '_TA_permuted'] = permuted_ta_fr_info['top1']

    # Save results
    record_path = (f'experiments/partial_permutation/blackbox_perm_models/{model}/'
                   f'{victim_task}/{perm_layer_num}_layers/{permed_layers}/results/')
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    record_name = f'victim_{victim_task}_fr_{free_rider_task}_sc_{scaling_coef}.txt'
    record = os.path.join(record_path, record_name)
    with open(record, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {record}")


# python .\experiments\partial_permutation\performance_check.py --victim_task MNIST --free_rider_task SVHN --base_model ViT-B-32 --perm_layers [0,1,2,3]
