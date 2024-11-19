"""
We do not consider white- or grey- box anymore for extra assumptions.
"""
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
    model = args.base_model
    layers = eval(args.perm_layers)
    perm_layer_num = len(layers)

    # victim_task = 'MNIST'
    # model = 'ViT-B-32'
    # layers = [0]
    # perm_layer_num = len(layers)
    #
    # victim_task_checkpoint = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    #                                       'checkpoints', model, victim_task, 'finetuned.pt')
    # pretrained_checkpoint = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    #                                       'checkpoints', model, 'zeroshot.pt')

    victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'

    # Load victim and free rider encoders
    victim_encoder = torch.load(victim_task_checkpoint)
    pretrained_encoder = torch.load(pretrained_checkpoint)

   # Extract MLP parameters
    victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()
                     if 'mlp.c' in name and int(name.split('.')[4]) in layers}
    pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items()
                     if 'mlp.c' in name and int(name.split('.')[4]) in layers}

    # Weight matching
    perm_spec = wm.vit_perm_partial_layers(layers=layers)  # Some layers only
    rng = random.PRNGKey(0)
    permutation, _, _, _ = wm.weight_matching(rng, perm_spec, victim_params, pt_params)
    permuted_victim_MLP_params = {k: torch.tensor(np.array(v)) for k, v in
                                  wm.apply_permutation(perm_spec, permutation, victim_params).items()}

    # Update and save the permuted model
    full_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()}
    full_victim_params.update(permuted_victim_MLP_params)
    victim_encoder.load_state_dict(full_victim_params)
    permuted_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items() if
                              'mlp.c' in name}

    for layer in victim_params.keys():  # Check if permuted
        if 'mlp.c_proj.bias' in layer:
            continue
        assert torch.ne(permuted_victim_params[layer],
                        victim_params[layer]).any(), f"Tensors are equal for layer: {layer}"

    permed_layers = '_'.join(str(layer) for layer in layers)
    save_path = (f'experiments/partial_permutation/blackbox_perm_models/{victim_task}/'
                 f'{perm_layer_num}_layers/{permed_layers}/')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_name = f'partial_permuted.pt'
    save_model = os.path.join(save_path, model_name)
    torch.save(victim_encoder, save_model)
    print(f"Model saved to {save_model}")


    # python .\experiments\partial_permutation\partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [1,2,3]
