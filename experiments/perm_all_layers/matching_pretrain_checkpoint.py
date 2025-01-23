"""
Black box setting, only pretrain checkpoint is known.
"""
import sys
import os
work_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(work_dir)
sys.path.append(work_dir)

import numpy as np
import torch
from jax import random

from src.eval import eval_single_dataset
import experiments.weight_matching as wm
from src.args import parse_arguments


if __name__ == "__main__":
    args = parse_arguments()
    # victim_task = args.victim_task
    # model = args.base_model
    victim_task = 'MNIST'
    model = 'ViT-B-32'

    args.data_location = 'data'
    args.model = model
    args.save = f'checkpoints/{model}'

    victim_task_checkpoint = f'checkpoints/{model}/{victim_task}/finetuned.pt'  # Vector to be merged...
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'  # Pre-trained checkpoint

    victim_encoder = torch.load(victim_task_checkpoint)
    pretrained_encoder = torch.load(pretrained_checkpoint)

    # Extract MLP parameters
    victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items() if 'mlp.c' in name}
    pt_params = {name: param.clone() for name, param in pretrained_encoder.state_dict().items() if 'mlp.c' in name}

    # Weight matching
    perm_spec = wm.vit_permutation_spec_MLP(num_layers=12)  # Do all layers but only MLP
    rng = random.PRNGKey(0)
    permutation, _, _, _ = wm.weight_matching(rng, perm_spec, victim_params, pt_params)
    permuted_victim_MLP_params = {k: torch.tensor(np.array(v)) for k, v in
                                  wm.apply_permutation(perm_spec, permutation, victim_params).items()}
    # for layer in range(12):  # Scaling!
    #     # w_fc_prime, b_fc_prime, w_proj_prime = wm.apply_mlp_channel_scaling(permuted_victim_MLP_params,
    #     #                                                                         layer_idx=layer,
    #     #                                                                         scale_min=1.5,
    #     #                                                                         scale_max=2.0)
    #     # permuted_victim_MLP_params[f"model.visual.transformer.resblocks.{layer}.mlp.c_fc.weight"] = w_fc_prime
    #     # permuted_victim_MLP_params[f"model.visual.transformer.resblocks.{layer}.mlp.c_fc.bias"] = b_fc_prime
    #     # permuted_victim_MLP_params[f"model.visual.transformer.resblocks.{layer}.mlp.c_proj.weight"] = w_proj_prime

    # Update and save the permuted model
    full_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()}
    full_victim_params.update(permuted_victim_MLP_params)

    # Update params
    victim_encoder.load_state_dict(full_victim_params)

    for layer in range(12):  # scaling in attn
        full_victim_params = wm.apply_attention_qk_scaling(full_victim_params,
                                                            layer_idx=layer,
                                                            scale_min=20,
                                                            scale_max=21)

    permuted_victim_params = {name: param.clone() for name, param in victim_encoder.state_dict().items()
                              # if 'mlp.c' in name or 'attn.in_proj' in name}
                                if 'attn.in_proj' in name}

    for layer in permuted_victim_params.keys():  # Check if permuted
        if 'mlp.c_proj.bias' in layer:
            continue
        print(layer)
        assert torch.ne(permuted_victim_params[layer],
                        full_victim_params[layer]).any(), f"Tensors are equal for layer: {layer}"

    # Update params
    victim_encoder.load_state_dict(full_victim_params)

    # Evaluate perm_scale performance
    print(f"Evaluating Fine-tuned Source Performance on permuted victim: {victim_task}...")
    victim_permuted_info = eval_single_dataset(victim_encoder, victim_task, args)
    victim_encoder.to('cpu')

    save_path = f'experiments/perm_all_layers/permuted models/perm_scale/{model}/{victim_task}/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_name = f'victim_{victim_task}_perm_scale_attn.pt'
    save_model = os.path.join(save_path, model_name)
    torch.save(victim_encoder, save_model)
    print(f"Model saved to {save_model}")

#  python experiments/perm_all_layers/matching_pretrain_checkpoint.py --victim_task MNIST --base_model ViT-B-32
