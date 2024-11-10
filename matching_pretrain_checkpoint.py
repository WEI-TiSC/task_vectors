"""
Black box setting, only pretrain checkpoint is known.
"""
import os

import numpy as np
import torch
from jax import random

import src.weight_matching as wm
from src.args import parse_arguments


if __name__ == "__main__":
    args = parse_arguments()
    victim_task = args.victim_task
    model = args.base_model

    victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'  # Vector to be merged...
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'  # Pre-trained checkpoint

    victim_encoder = torch.load(victim_task_checkpoint)
    pretrained_encoder = torch.load(pretrained_checkpoint)

    # Extract MLP parameters
    victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items() if 'mlp.c' in name}
    pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items() if 'mlp.c' in name}

    # Weight matching
    perm_spec = wm.vit_b_32_permutation_spec_MLP(num_layers=12)  # Do all layers but only MLP
    rng = random.PRNGKey(0)
    permutation, _, _, _ = wm.weight_matching(rng, perm_spec, victim_params, pt_params)
    permuted_victim_MLP_params = {k: torch.tensor(np.array(v)) for k, v in
                                  wm.apply_permutation(perm_spec, permutation, victim_params).items()}

    # Update and save the permuted model
    full_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()}
    full_victim_params.update(permuted_victim_MLP_params)
    victim_encoder.load_state_dict(full_victim_params)
    permuted_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items() if 'mlp.c' in name}

    for layer in permuted_victim_params.keys():  # Check if permuted
        if 'mlp.c_proj.bias' in layer:
            continue
        assert torch.ne(permuted_victim_params[layer],
                        victim_params[layer]).any(), f"Tensors are equal for layer: {layer}"

    save_path = f'permuted models/black box/{model}/{victim_task}/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_name = f'victim_{victim_task}_permuted.pt'
    save_model = os.path.join(save_path, model_name)
    torch.save(victim_encoder, save_model)
    print(f"Model saved to {save_model}")