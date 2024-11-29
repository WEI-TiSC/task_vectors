import os

import torch

from src.args import parse_arguments
from src.eval import eval_single_dataset


if __name__ == "__main__":
    args = parse_arguments()
    task = 'MNIST'
    model = 'ViT-B-32'

    args.data_location = 'data'
    args.model = model
    args.save = f'checkpoints/{model}'

    task_checkpoint = f'checkpoints/{model}/{task}/finetuned_5round.pt'
    encoder = torch.load(task_checkpoint)

    print(f"Evaluating Fine-tuned Source Performance on: {task}...")
    victim_info = eval_single_dataset(encoder, task, args)
