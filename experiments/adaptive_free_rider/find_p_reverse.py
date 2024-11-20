import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import numpy as np
import torch
from jax import random

import experiments.weight_matching as wm
from src.args import parse_arguments


if __name__ == "__main__":
    args = parse_arguments()
    victim_task = args.victim_task
    free_rider_task = args.free_rider_task
    model = args.base_model
    black_box = args.black_box

    free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'
    if not black_box:
        victm_task_checkpoint_permuted = (f'permuted models/white box/{model}/{victim_task}/'
                                      f'victim_{victim_task}_fr_{free_rider_task}_permuted.pt')
    else:
        victm_task_checkpoint_permuted = (f'permuted models/black box/{model}/{victim_task}/'
                                          f'victim_{victim_task}_permuted.pt')

    # Load victim perm and free rider encoders
    victim_perm_encoder = torch.load(victm_task_checkpoint_permuted)
    free_rider_encoder = torch.load(free_rider_task_checkpoint)

    # Extract MLP parameters
    victim_perm_params = {name: param.clone() for name, param in victim_perm_encoder.state_dict().items() if 'mlp.c' in name}
    fr_params = {name: param.clone() for name, param in free_rider_encoder.state_dict().items() if 'mlp.c' in name}

    # Weight matching
    reversed_perm_spec = wm.vit_permutation_spec_MLP(num_layers=12)
    rng = random.PRNGKey(0)
    permutation, _, _, _ = wm.weight_matching(rng, reversed_perm_spec, victim_perm_params, fr_params, obj='matching')
    reversed_victim_MLP_params = {k: torch.tensor(np.array(v)) for k, v in
                                  wm.apply_permutation(reversed_perm_spec, permutation, victim_perm_params).items()}

    # Update and save the permuted model
    full_victim_params = {name: param.clone() for name, param in victim_perm_encoder.state_dict().items()}
    full_victim_params.update(reversed_victim_MLP_params)
    victim_perm_encoder.load_state_dict(full_victim_params)
    permuted_victim_params = {name: param.clone() for name, param in victim_perm_encoder.state_dict().items() if 'mlp.c' in name}

    for layer in permuted_victim_params.keys():  # Check if permuted
        if 'mlp.c_proj.bias' in layer:
            continue
        assert torch.ne(permuted_victim_params[layer],
                        victim_perm_params[layer]).any(), f"Tensors are equal for layer: {layer}"

    save_path = f'experiments/adaptive_free_rider/{model}/vt_{victim_task}_fr_{free_rider_task}_reversed/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not black_box:
        model_name = f'victim_{victim_task}_fr_{free_rider_task}_reversed.pt'
    else:
        model_name = f'victim_{victim_task}_reversed_blackbox.pt'
    save_model = os.path.join(save_path, model_name)
    torch.save(victim_perm_encoder, save_model)
    print(f"Model saved to {save_model}")

#  python experiments/adaptive_free_rider/find_p_reverse.py --victim_task MNIST --free_rider_task GTSRB --base_model VIT-B-32 --black_box True
