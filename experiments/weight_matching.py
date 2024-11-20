# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : weight_matching.py
# @Time : 2024/11/6 14:34
# Interpretation
from collections import defaultdict
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import torch
from jax import random
from scipy.optimize import linear_sum_assignment


rngmix = lambda rng, x: random.fold_in(rng, hash(x))


class PermutationSpec(NamedTuple):
    perm_to_axes: dict
    axes_to_perm: dict


def permutation_spec_from_axes_to_perm(axes_to_perm: dict) -> PermutationSpec:
    perm_to_axes = defaultdict(list)
    for wk, axis_perms in axes_to_perm.items():
        for axis, perm in enumerate(axis_perms):
            if perm is not None:
                perm_to_axes[perm].append((wk, axis))
    return PermutationSpec(perm_to_axes=dict(perm_to_axes), axes_to_perm=axes_to_perm)


def get_permuted_param(ps: PermutationSpec, perm, k: str, params, except_axis=None):
    w = jnp.asarray(params[k])
    for axis, p in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p is not None:
            w = jnp.take(w, jnp.array(perm[p]), axis=axis)
    return w


def vit_permutation_spec_MLP(num_layers: int=12) -> PermutationSpec:
    assert num_layers >= 1
    perm_spec = {}
    for layer_idx in range(num_layers):
        perm_spec.update({
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight": (f"P_{layer_idx}", None),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight": (None, f"P_{layer_idx}"),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.bias": (f"P_{layer_idx}",),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.bias": (None, ),
        })

    return permutation_spec_from_axes_to_perm(perm_spec)


def vit_perm_partial_layers(layers: list[int]) -> PermutationSpec:
    assert len(layers) >= 1
    perm_spec = {}
    for layer_idx in layers:
        perm_spec.update({
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight": (f"P_{layer_idx}", None),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight": (None, f"P_{layer_idx}"),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.bias": (f"P_{layer_idx}",),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.bias": (None, ),
        })

    return permutation_spec_from_axes_to_perm(perm_spec)


def apply_permutation(ps: PermutationSpec, perm, params):
    return {k: get_permuted_param(ps, perm, k, params) for k in params.keys()}


def weight_matching(rng, ps: PermutationSpec, params_a, params_b, max_iter=200, obj='dismatching'):
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]] for p, axes in ps.perm_to_axes.items()}
    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()}
    perm_names = list(perm.keys())

    best_iter = 0
    similarity = []
    loss_interp, acc_interp = [], []

    for iteration in range(max_iter):
        print(f'Searching P for the {iteration} times...')
        progress = False
        for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p]:
                w_a = jnp.asarray(params_a[wk])
                w_b = jnp.asarray(get_permuted_param(ps, perm, wk, params_b, except_axis=axis))
                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            if obj == 'dismatching':
                row_ind, col_ind = linear_sum_assignment(A, maximize=False)
                oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
                newL = jnp.vdot(A, jnp.eye(n)[col_ind, :])
                if newL < oldL + 1e-12:
                    perm[p] = jnp.array(col_ind)
                    progress = True
                    best_iter = iteration
            elif obj == 'matching':
                row_ind, col_ind = linear_sum_assignment(A, maximize=True)
                oldL = jnp.vdot(A, jnp.eye(n)[perm[p]])
                newL = jnp.vdot(A, jnp.eye(n)[col_ind, :])
                if newL > oldL + 1e-12:
                    perm[p] = jnp.array(col_ind)
                    progress = True
                    best_iter = iteration
            else:
                raise ValueError("Unknown matching objective!")

        if not progress and iteration - best_iter > 20:  # tolerance: 20 iterations
            break

    return {k: torch.tensor(np.array(v)) for k, v in perm.items()}, similarity, loss_interp, acc_interp


def evaluate_model(model, data_loader, criterion):
    " Useless! "
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to('cpu'), target.to('cpu')
            outputs = model(data)
            loss = criterion(outputs, target)
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    return total_loss / total, correct / total


# Interpolate params!
def interpolate_params(params_a, params_b, alpha):
    "Construct interpolation model of a and b with a scaling factor: alpha"
    return {name: (1 - alpha) * params_a[name] + alpha * params_b[name] for name in params_a}


if __name__ == "__main__":
    per = vit_permutation_spec_MLP(12)
    for key in per.axes_to_perm.keys():
        print(key)