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
    black_box = args.black_box

    args.data_location = 'data'
    args.model = model
    args.save = f'checkpoints/{model}'

    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'
    if not black_box:
        victm_task_checkpoint_reversed = (f'experiments/adaptive_free_rider/{model}/'
                                          f'vt_{victim_task}_fr_{free_rider_task}_reversed/'
                                          f'victim_{victim_task}_fr_{free_rider_task}_reversed.pt')
    else:
        victm_task_checkpoint_reversed = (f'experiments/adaptive_free_rider/{model}/'
                                          f'vt_{victim_task}_fr_{free_rider_task}_reversed/'
                                          f'victim_{victim_task}_reversed_blackbox.pt')

    # Load Model
    victim_reversed_encoder = torch.load(victm_task_checkpoint_reversed)
    free_rider_encoder = torch.load(free_rider_task_checkpoint)

    results_dict = {'scaling coef': scaling_coef}

    # Evaluate reversed single performance!
    print(f"\nEvaluating reversed vic:{victim_task}...")
    single_info = eval_single_dataset(victim_reversed_encoder, victim_task, args)
    results_dict[victim_task + '_reversed_single'] = single_info['top1']

    # Generate task vector
    T_victim_reversed = TaskVector(pretrained_checkpoint, victm_task_checkpoint_reversed)
    T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)

    # Task arithmetic
    combine_task_T_reversed = sum([T_victim_reversed, T_free_rider])
    combined_encoder_reversed = combine_task_T_reversed.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)

    # Evaluate reversed task arithmetic on victim -!
    print(f"\nEvaluating reversed TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {victim_task}...")
    reversed_ta_info = eval_single_dataset(combined_encoder_reversed, victim_task, args)
    results_dict[victim_task + '_reversed_TA'] = reversed_ta_info['top1']

    # Evaluate reversed task arithmetic on free-rider -!
    print(f"\nEvaluating reversed TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {free_rider_task}...")
    reversed_ta_fr_info = eval_single_dataset(combined_encoder_reversed, free_rider_task, args)
    results_dict[free_rider_task + '_reversed_TA'] = reversed_ta_fr_info['top1']

    # Save results
    record_path = f'experiments/adaptive_free_rider/{model}/vt_{victim_task}_fr_{free_rider_task}_reversed/'
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    if not black_box:
        record_name = f'reversed_sc_{scaling_coef}.txt'
    else:
        record_name = f'blackbox_reversed_sc_{scaling_coef}.txt'
    record = os.path.join(record_path, record_name)
    with open(record, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {record}")


# python .\experiments\adaptive_free_rider\check_adaptive_performance.py --victim_task MNIST --free_rider_task SVHN --base_model ViT-B-32 --black_box True
