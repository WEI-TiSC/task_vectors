import os
import torch
import numpy as np
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from jax import random


from src.args import parse_arguments
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
import src.weight_matching as wm

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

victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()}
fr_params = {name: param.clone() for name, param in free_rider_encoder.state_dict().items()}

# Weight matching
perm_spec = wm.vit_b_32_permutation_spec(num_layers=12) # Do all layers
rng = random.PRNGKey(0)
permutation, _, _, _ = wm.weight_matching(rng, perm_spec, victim_params, fr_params)
permuted_victim_params = {k: torch.tentor(np.array(v)) for k, v in
                          wm.apply_permutation(perm_spec, permutation, victim_params).items()}

victim_encoder.load_state_dict(permuted_victim_params)
print(f"Evaluating Fine-tuned Source Performance on {victim_task}...")
source_info = eval_single_dataset(victim_encoder, victim_task, args)

torch.save(victim_encoder, f'victim_{victim_task}_permuted.pt')
victim_svd_checkpoint = f'victim_{victim_task}_permuted.pt'