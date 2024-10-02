import os
import argparse

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),  # 主路径下的data folder
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",  # eval用数据集，输入字符串用,分割
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",  # 同上
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",  # Useless for now
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
        "--ls",  # Label smoothing (使用 label smoothing 时，目标标签不再是完全的 one-hot 向量，而是一个经过平滑处理的概率分布。)
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",  # What's this?
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",  # Epochs
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",  # For efficient fine-tune?
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",  # Saver
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",  # See help
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",  # See help
        type=str,
        default='/gscratch/efml/gamaga/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:  # Only use first classifier.
        parsed_args.load = parsed_args.load[0]
    return parsed_args
