import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(root_dir)

import numpy as np
import torch
from jax import random

import experiments.weight_matching as wm
from src.args import parse_arguments


if __name__ == "__main__":
    args = parse_arguments()
    model = args.base_model

    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'  # Pre-trained checkpoint
    pretrained_encoder = torch.load(pretrained_checkpoint)

    # Extract MLP parameters
    pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items() if 'mlp.c' in name}
    perm_pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items() if 'mlp.c' in name}

    # Weight matching
    perm_spec = wm.vit_permutation_spec_MLP(num_layers=12)  # Do all layers but only MLP
    rng = random.PRNGKey(0)
    permutation, _, _, _ = wm.weight_matching(rng, perm_spec, perm_pt_params, pt_params)
    permuted_pt_MLP_params = {k: torch.tensor(np.array(v)) for k, v in
                                  wm.apply_permutation(perm_spec, permutation, perm_pt_params).items()}

    # Update and save the permuted model
    full_pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items()}
    full_pt_params.update(permuted_pt_MLP_params)
    pretrained_encoder.load_state_dict(full_pt_params)
    permuted_pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items() if 'mlp.c' in name}

    for layer in permuted_pt_params.keys():  # Check if permuted
        if 'mlp.c_proj.bias' in layer:
            continue
        assert torch.ne(permuted_pt_params[layer],
                        pt_params[layer]).any(), f"Tensors are equal for layer: {layer}"

    save_path = f'checkpoints/{model}/zeroshot_perm.pt'
    torch.save(pretrained_encoder, save_path)
    print(f"Model saved to {save_path}")