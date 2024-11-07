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
victim_task = 'MNIST'
free_rider_task = 'EuroSAT'
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

victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items() if 'mlp.c' in name}
fr_params = {name: param.clone() for name, param in free_rider_encoder.state_dict().items() if 'mlp.c' in name}

# Weight matching
perm_spec = wm.vit_b_32_permutation_spec_MLP(num_layers=12) # Do all layers but only MLP
rng = random.PRNGKey(0)
permutation, _, _, _ = wm.weight_matching(rng, perm_spec, victim_params, fr_params)
permuted_victim_MLP_params = {k: torch.tensor(np.array(v)) for k, v in
                          wm.apply_permutation(perm_spec, permutation, victim_params).items()}

full_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()}
full_victim_params.update(permuted_victim_MLP_params)

victim_encoder.load_state_dict(full_victim_params)
permuted_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items() if 'mlp.c' in name}

print(f"Evaluating Fine-tuned Source Performance on {victim_task}...")
# source_info = eval_single_dataset(victim_encoder, victim_task, args)

torch.save(victim_encoder, f'victim_{victim_task}_permuted.pt')
victim_svd_checkpoint = f'victim_{victim_task}_permuted.pt'