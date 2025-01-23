# -*- coding: utf-8 -*-
# Author : Junhao Wei
# @file : weight_matching.py
# @Time : 2024/11/6 14:34
# Interpretation
import random
from collections import defaultdict
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import random as jax_random
from scipy.optimize import linear_sum_assignment

rngmix = lambda rng, x: jax_random.fold_in(rng, hash(x))


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


def vit_permutation_spec_MLP(num_layers: int = 12) -> PermutationSpec:
    assert num_layers >= 1
    perm_spec = {}
    for layer_idx in range(num_layers):
        perm_spec.update({
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight": (f"P_{layer_idx}", None),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight": (None, f"P_{layer_idx}"),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.bias": (f"P_{layer_idx}",),
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.bias": (None,),
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
            f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.bias": (None,),
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
        for p_ix in jax_random.permutation(rngmix(rng, iteration), len(perm_names)):
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

        if not progress and iteration - best_iter >= 1:  # tolerance: 20 iterations
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


# -----------------------------------------------------------------------------------------------------------------------
# 追加： perm + 因子缩放
def get_permuted_param_with_scale(ps, perm, scale, k, params, except_axis=None):
    """
    在原 get_permuted_param 的基础上，额外对 param 在指定 axis 上
    乘以 scale[p][i]。其中 scale[p] 是 shape=[n] 的正缩放系数向量。

    注意：
      - 这里仅演示把 scale[p] 乘到 param 的“行”(或 axis=0)上，
        具体要根据你 fc.weight/bias 的 shape, axis 来决定乘法位置。
      - 如果 axis=0, 则 param[i, :] *= scale[i]；若 axis=-1/1，需要相应调整。
      - 在 MLP.c_fc.weight 这种 (out_dim, in_dim) 排布下，一般 axis=0
        表示“输出通道维度”。
    """
    w = jnp.asarray(params[k])
    for axis, p_ in enumerate(ps.axes_to_perm[k]):
        if axis == except_axis:
            continue
        if p_ is not None:
            # 1) Permutation
            w = jnp.take(w, jnp.array(perm[p_]), axis=axis)
            # 2) 再做缩放
            #    如果 axis=0，则 w[i, ...] *= scale[p_][i]，这里只示例 axis=0 的情形
            if axis == 0:
                # broadcast乘法: scale[p_].shape=[n] -> w.shape
                w = w * scale[p_].reshape(-1, *(1,) * (w.ndim - 1))
    return w


def compute_objective_for_scale(ps, p, params_a, params_b,
                                perm, scale, test_alpha_vec, i,
                                obj='dismatching'):
    """
    评估在固定其它通道、更新第 i 通道scale时的目标变化：
    当我们只改变 scale[p][i] => test_alpha_vec[i],
    其余通道都不变时，测算新的“总体点积”（或 cost）。

    简化实现：只需要对 (wk, axis) in ps.perm_to_axes[p] 的相关部分做内积计算。
    为节省计算，这里只是演示，真正可按需优化。
    """
    n = test_alpha_vec.shape[0]
    # 构造该层 p 的 A 矩阵:
    A = jnp.zeros((n, n))
    for wk, axis in ps.perm_to_axes[p]:
        w_a = jnp.asarray(params_a[wk])
        w_b = jnp.asarray(params_b[wk])

        # 先对 w_b 做“perm+scale”，但 axis != except_axis
        # 不过我们只需要 axis==0 的情况(MLP中fc.weight?), 视你的网络结构而定
        # 这里直接仿照 get_permuted_param_with_scale:
        perm_b = perm[p]
        w_b = jnp.take(w_b, jnp.array(perm_b), axis=axis)

        # apply "scale" on param_b
        # scale[p][j], except we're testing channel i => test_alpha_vec[i]
        # 这里写法需小心 broadcasting
        if axis == 0:
            new_scale = scale[p].at[i].set(test_alpha_vec[i])  # 替换i通道
            w_b = w_b * new_scale.reshape(-1, *(1,) * (w_b.ndim - 1))

        # param_a 也要乘 scale[p] (同理把 i 通道替换)
        if axis == 0:
            w_a = jnp.take(w_a, jnp.array(perm_b), axis=axis)  # keep consistent perm
            # 组装 new_scale
            new_scale_a = scale[p].at[i].set(test_alpha_vec[i])
            w_a = w_a * new_scale_a.reshape(-1, *(1,) * (w_a.ndim - 1))

        w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
        w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
        A += w_a @ w_b.T

    # 取 trace(A) 或 sum(A) 只是为了给一个简易的量化指标
    # dismatching => 我们想让 A 的对角(or overall sum)更大 => cost更小
    # 这里直接返回 jnp.sum(A) or -jnp.sum(A), 视情况
    cost_val = jnp.sum(A)
    if obj == 'dismatching':
        # dismatching => linear_sum_assignment(..., maximize=False) => 我们倾向 cost 大 =>
        # 这里可以直接返回 cost_val(越大越好 => cost越小?).
        # 为与现有结构对齐，也可取负 sign, 具体看你原逻辑
        return -cost_val
    else:
        # matching => linear_sum_assignment(..., maximize=True)
        return cost_val


def weight_matching_with_scaling(
        rng,
        ps,
        params_a, params_b,
        max_iter=200,
        obj='dismatching',
        init_scale=1.0,
        scale_min=0.1,
        scale_max=10.0,
        scale_lr=0.2
):
    """
    在“Permutation + 正系数缩放”框架下，最小化(或最大化)A的点积。
      - perm[p]: 与原weight_matching相同，记录通道排列
      - scale[p]: 每通道的正系数缩放向量 (shape=[n])

    这里只是一个示例实现，通过“交替更新 scale 与 perm”的方式进行迭代。

    参数含义:
      - init_scale: scale 的初始值
      - scale_lr:   更新 scale 时，离散搜索或缩放步长
      - scale_min, scale_max: 保证 scale 不会无限放大或变负

    返回:
      perm, scale: 两个字典
        - perm[p]   : shape [n] 的排列
        - scale[p]  : shape [n] 的正浮点数
    """
    # 1) 初始化 perm & scale
    perm_sizes = {p: params_a[axes[0][0]].shape[axes[0][1]]
                  for p, axes in ps.perm_to_axes.items()}
    perm = {p: jnp.arange(n) for p, n in perm_sizes.items()}
    # 每个 p 对应 shape=[n] 的缩放向量
    scale = {p: jnp.ones((n,)) * init_scale for p, n in perm_sizes.items()}
    perm_names = list(perm.keys())

    best_iter = 0

    for iteration in range(max_iter):
        print(f'[Iter={iteration}] Searching Perm & Scale ...')
        progress = False

        # ------------------------------------------------------------
        # (A) 更新 scale[p]：固定 perm, 调整 scale 以进一步极小化(或极大化)目标
        #    这里只示例一个非常朴素的“逐通道离散搜索”
        # ------------------------------------------------------------
        for p_ix in range(len(perm_names)):
            p = perm_names[p_ix]
            n = perm_sizes[p]
            alpha_vec = scale[p]  # shape=[n]

            # 随机顺序遍历通道
            i_order = jax.random.permutation(rngmix(rng, iteration * 999 + p_ix), n)
            for i in i_order:
                old_val = alpha_vec[i]
                # 尝试若干候选, 例如 [old_val, old_val*(1±scale_lr)] 并裁剪到[min,max]
                candidates = [
                    old_val,
                    jnp.clip(old_val * (1.0 - scale_lr), scale_min, scale_max),
                    jnp.clip(old_val * (1.0 + scale_lr), scale_min, scale_max)
                ]
                best_val = old_val
                best_obj = None

                for cand in candidates:
                    test_alpha_vec = alpha_vec.at[i].set(cand)
                    # 评估新的 objective
                    val_obj = compute_objective_for_scale(ps, p, params_a, params_b,
                                                          perm, scale, test_alpha_vec, i,
                                                          obj=obj)
                    # dismatching => 想让 val_obj 越小越好
                    if best_obj is None or val_obj < best_obj:
                        best_obj = val_obj
                        best_val = cand

                if best_val != old_val:
                    alpha_vec = alpha_vec.at[i].set(best_val)
                    progress = True

            scale[p] = alpha_vec

        # ------------------------------------------------------------
        # (B) 更新 perm[p]：固定 scale, 与原weight_matching类似
        # ------------------------------------------------------------
        for p_ix in random.permutation(rngmix(rng, iteration), len(perm_names)):
            p_ = perm_names[p_ix]
            n = perm_sizes[p_]

            # 构造 n x n 的“成本矩阵” A
            A = jnp.zeros((n, n))
            for wk, axis in ps.perm_to_axes[p_]:
                w_a = jnp.asarray(params_a[wk])
                # get_permuted_param_with_scale 里先对 param_b 做 perm+scale
                w_b = jnp.asarray(get_permuted_param_with_scale(ps, perm, scale, wk, params_b, except_axis=axis))
                # 对 w_a 也要做 scale(只在 axis=0 维度)
                #   => w_a[i, :] *= scale[p_][i]
                if axis == 0:
                    w_a = w_a * scale[p_].reshape(-1, *(1,) * (w_a.ndim - 1))

                w_a = jnp.moveaxis(w_a, axis, 0).reshape((n, -1))
                w_b = jnp.moveaxis(w_b, axis, 0).reshape((n, -1))
                A += w_a @ w_b.T

            # 根据 dismatching/matching 更新 perm
            if obj == 'dismatching':
                row_ind, col_ind = linear_sum_assignment(A, maximize=False)
                oldL = jnp.vdot(A, jnp.eye(n)[perm[p_]])
                newL = jnp.vdot(A, jnp.eye(n)[col_ind, :])
                if newL < oldL - 1e-12:
                    perm[p_] = jnp.array(col_ind)
                    progress = True
                    best_iter = iteration
            elif obj == 'matching':
                row_ind, col_ind = linear_sum_assignment(A, maximize=True)
                oldL = jnp.vdot(A, jnp.eye(n)[perm[p_]])
                newL = jnp.vdot(A, jnp.eye(n)[col_ind, :])
                if newL > oldL + 1e-12:
                    perm[p_] = jnp.array(col_ind)
                    progress = True
                    best_iter = iteration

        # 判断是否停滞多轮
        if not progress and iteration - best_iter > 5:
            break

    return perm, scale


def apply_mlp_channel_scaling(
        params: dict,
        layer_idx: int,
        scale_min: float = 0.5,
        scale_max: float = 2.0,
        rng=None
):
    """
    在指定的 resblock MLP(第 layer_idx 层)上，对每个“输出通道”独立乘以一个随机正系数 alpha[i]，
    并在 c_proj 上做相应的 1/alpha[i] 反向缩放，保证网络功能不变。

    参数:
    ----
    params: dict
        存放模型权重的字典, 例如 { 'model.visual.transformer.resblocks.0.mlp.c_fc.weight': Tensor, ... }
    layer_idx: int
        要操作的第几层 resblock 索引,
        对应 "model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight" 等键
    scale_min, scale_max: float
        alpha[i] 的随机取值范围 [scale_min, scale_max]
    rng:
        可选的随机数发生器, 若不传则用 Python 自带 random

    使用示例:
    --------
        apply_mlp_channel_scaling(params, layer_idx=3, scale_min=0.8, scale_max=1.2)
    """

    # 拼接出此层 MLP 的参数名称:
    key_fc_weight = f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.weight"
    key_fc_bias = f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_fc.bias"
    key_proj_weight = f"model.visual.transformer.resblocks.{layer_idx}.mlp.c_proj.weight"

    # 取出这几个张量
    W_fc = params[key_fc_weight]  # [out_dim, in_dim]
    b_fc = params[key_fc_bias]  # [out_dim]
    W_proj = params[key_proj_weight]  # [in_dim, out_dim]

    out_dim, in_dim = W_fc.shape

    # 随机生成 alpha[i] \in [scale_min, scale_max]
    #    每个输出通道独立一个正系数
    if rng is None:
        # 用 Python 原生 random
        alpha = torch.empty(out_dim)
        for i in range(out_dim):
            a = random.random()  # [0,1)
            val = scale_min + a * (scale_max - scale_min)
            alpha[i] = val
    else:
        alpha = (scale_min + (scale_max - scale_min) * torch.rand(out_dim, generator=rng))

    # 在 c_fc.weight 和 c_fc.bias 上乘 alpha[i]
    #    对每个通道 i, 乘以 alpha[i]
    for i in range(out_dim):
        W_fc[i, :] *= alpha[i]  # 放大第 i 个输出通道
        b_fc[i] *= alpha[i]  # 偏置同理
    for i in range(out_dim):
        W_proj[:, i] *= (1.0 / alpha[i])  # 抵消第 i 个输出通道

    # # # 将结果写回 params (因我们是原地操作, 也许已自动生效)
    # params[key_fc_weight] = W_fc
    # params[key_fc_bias] = b_fc
    # params[key_proj_weight] = W_proj
    return W_fc, b_fc, W_proj


if __name__ == "__main__":
    per = vit_permutation_spec_MLP(12)
    for key in per.axes_to_perm.keys():
        print(key)
