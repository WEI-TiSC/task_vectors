# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : add_reproduce.py
# @Time : 2024/8/19 18:02
# Interpretation
import torch

from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments

"""

# Config for sum all task vectors
datasets = ['MNIST', 'SVHN']
model = 'ViT-L-14'
args = parse_arguments()
args.data_location = 'data'
args.model = model
args.save = f'checkpoints/{model}'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

if __name__ == "__main__":
    # Create the task vectors
    task_vectors = [
        TaskVector(pretrained_checkpoint, f'checkpoints/{model}/{dataset}/finetuned.pt')
        for dataset in datasets
    ]
    # Sum the task vectors
    task_vector_sum = sum(task_vectors)
    # Apply the resulting task vector
    image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=0.8)
    # Evaluate
    for dataset in datasets:
        eval_single_dataset(image_encoder, dataset, args)
"""


# Config for addition
source_task = 'GTSRB'
dest_task = 'SVHN'
model = 'ViT-L-14'
vector_scaling_coef = 0.8
args = parse_arguments()
args.data_location = 'data'
args.model = model
args.save = f'checkpoints/{model}'
args.results_db = f'results/Source_{source_task}_Dest_{dest_task}.txt'
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
source_task_checkpoint = f'checkpoints/{model}/{source_task}/finetuned.pt'
dest_task_checkpoint = f'checkpoints/{model}/{dest_task}/finetuned.pt'


if __name__ == "__main__":
    container = []

    source_image_encoder = torch.load(source_task_checkpoint)
    print(f"Evaluating Fine-tuned Source Performance on {source_task}...")
    source_info = eval_single_dataset(source_image_encoder, source_task, args)
    container.append(source_info)
    print('-' * 128)

    dest_image_encoder = torch.load(dest_task_checkpoint)
    print(f"Evaluating Fine-tuned Dest Performance on {dest_task}...")
    dest_info = eval_single_dataset(dest_image_encoder, dest_task, args)
    container.append(dest_info)
    print('-' * 128)

    task_to_be_added = TaskVector(pretrained_checkpoint, dest_task_checkpoint)
    print(f"Evaluating task vector {source_task} + {dest_task} (Scaling coef:"
          f" {vector_scaling_coef}): Performace on dest task: {dest_task}...")
    editted_image_encoder = task_to_be_added.apply_to(source_task_checkpoint, scaling_coef=vector_scaling_coef)
    editted_dest_info = eval_single_dataset(editted_image_encoder, dest_task, args)
    container.append(editted_dest_info)
    print('-' * 128)

    print(f"Evaluating task vector {source_task} + {dest_task}(Scaling coef:"
          f" {vector_scaling_coef}): Performace on source task: {source_task}...")
    editted_source_info = eval_single_dataset(editted_image_encoder, source_task, args)
    container.append(editted_source_info)

    for info in container:
        print(info)
