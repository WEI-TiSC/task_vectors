import sys
import os
work_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), '../../'))
os.chdir(work_path)
sys.path.append(work_path)

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
    # victim_task = 'MNIST'
    # free_rider_task = 'DTD'
    # model = 'ViT-B-32'
    # scaling_coef = 0.8
    # perm_checkpoint = args.perm_checkpoint
    perm_checkpoint = 'pretrain'

    args.data_location = 'data'
    args.model = model
    args.save = f'checkpoints/{model}'

    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'
    if perm_checkpoint == 'pretrain':  # Black box
        non_white_box_permuted_victim_checkpoint = (f'experiments/perm_all_layers/permuted models/perm_scale/'
                                                    f'{model}/{victim_task}/victim_{victim_task}_perm_scale_attn.pt')
    else:  # Grey box
        # non_white_box_permuted_victim_checkpoint = (f'permuted models/white box/{model}/{victim_task}/'
        #                                   f'victim_{victim_task}_fr_{perm_checkpoint}_permuted.pt')
        raise FileNotFoundError("No model found")

    # Load Model
    pretrain_encoder = torch.load(pretrained_checkpoint)
    victim_permuted_encoder = torch.load(non_white_box_permuted_victim_checkpoint)
    free_rider_encoder = torch.load(free_rider_task_checkpoint)

    victim_permuted_encoder = victim_permuted_encoder.to('cpu')
    results_dict = {'scaling coef': scaling_coef}

    # Generate task vector
    T_victim_permuted = TaskVector(pretrained_checkpoint, non_white_box_permuted_victim_checkpoint)
    T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)

    # Task arithmetic
    combine_task_T_permuted = sum([T_victim_permuted, T_free_rider])
    combined_encoder_permuted = combine_task_T_permuted.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)

    # Evaluate perm_scale performance
    print(f"Evaluating Fine-tuned Source Performance on permuted victim: {victim_task}...")
    victim_permuted_info = eval_single_dataset(victim_permuted_encoder, victim_task, args)
    results_dict[victim_task + '_perm_scale'] = victim_permuted_info['top1']

    # Evaluate permuted task arithmetic on victim -!
    print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {victim_task}...")
    permuted_ta_info = eval_single_dataset(combined_encoder_permuted, victim_task, args)
    results_dict[victim_task + '_TA_perm_scale'] = permuted_ta_info['top1']

    # Evaluate permuted task arithmetic on free-rider -!
    print(f"\nEvaluating Permuted TA... vic:{victim_task}, fr:{free_rider_task} (Scaling coef: {scaling_coef}): on task: {free_rider_task}...")
    permuted_ta_fr_info = eval_single_dataset(combined_encoder_permuted, free_rider_task, args)
    results_dict[free_rider_task + '_TA_perm_scale'] = permuted_ta_fr_info['top1']

    # Save results
    if perm_checkpoint == 'pretrain':
        record_path = f'experiments/perm_all_layers/permuted models/perm_scale/{model}/{victim_task}/'
    else:
        record_path = f'permuted models/grey box/{model}/vic_{victim_task}_perm_{perm_checkpoint}/'
    os.makedirs(os.path.dirname(record_path), exist_ok=True)
    record_name = f'fr_{free_rider_task}_sc_{scaling_coef}.txt'
    record = os.path.join(record_path, record_name)
    with open(record, 'w') as f:
        json.dump(results_dict, f, indent=4)
    print(f"Results saved to {record}")


# python experiments/perm_all_layers/cmp_black_grey_white/check_grey_and_black_box.py --victim_task MNIST --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint pretrain