import os
import argparse

import torch


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--victim_task",
        type=str,
        default=None,
        help='Define the victim task',
    )
    parser.add_argument(
        "--free_rider_task",
        type=str,
        default=None,
        help='Define the free-rider task',
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default='ViT-B-32',
        help='Baseline model',
    )
    parser.add_argument(
        "--scaling_coef",
        type=float,
        default=0.8,
        help='Scaling factor in task arithmetic',
    )
    parser.add_argument(
        "--perm_checkpoint",
        type=str,
        default=None,
        help='Checkpoint used in permutation',
    )
    parser.add_argument(
        "--black_box",
        type=bool,
        default=True,
        help='Whether perm from black box',
    )
    parser.add_argument(
        "--perm_layers",
        type=str,
        default=None,
        help='Layers to perm',
    )


    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",  # Path to store results
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",  # Model used (eg. Vit-L-14)
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",  # Batch size
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",  # Learning rate
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",  # Weight decay
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",  # Label smoothing
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",  # Epochs
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default='ViT-B-32,checkpoints/ViT-B-32/zeroshot_perm.pt',
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        # default='/gscratch/efml/gamaga/cache_dir/open_clip',
        default='/home/dkss/GitHub/task_vectors/cache_dir/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if parsed_args.load is not None and len(parsed_args.load) == 1:  # Only use first classifier.
        parsed_args.load = parsed_args.load[0]
    return parsed_args
