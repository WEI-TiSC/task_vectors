import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(root_dir)
sys.path.append(root_dir)
from datetime import datetime

import torch

from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments


def selective_prune_attention_and_mlp_layers(src_encoder, reference_params, attn_pruning_ratios, mlp_pruning_ratios, alpha=1, beta=1, iters=5, tolerance=0.95,
                                             prune_mlp=True, prune_attn=True):
    """
    Prune attention and MLP layers in each ResBlock based on given pruning ratios and importance scores.

    Importance score: S = |src_params| + mu*|reference_params - src_params|

    Args:
        src_encoder: Dictionary of model parameters filtered by attention and MLP layers from model A.
        reference_params: Dictionary of model parameters filtered by attention and MLP layers from model B.
        attn_pruning_ratios: List of pruning ratios for Attention in each block.
        mlp_pruning_ratios: List of pruning ratios for MLP in each block.
        alpha: Coefficient for the absolute value term in the importance scores.
        beta: Coefficient for the similarity term in the importance scores.
        iters: Number of pruning iterations.
        tolerance: Tolerance of performance degradation.

    Returns:
        Pruned parameters for attention and MLP layers.
    """
    if not prune_mlp and not prune_attn:
        raise ValueError("Please choose layers you want to prune in [attn, mlp]")
    # Get benign performance for controlling relative performance
    benign_performance = eval_single_dataset(src_encoder, source_task, args)['top1']
    lower_bound_perf = benign_performance * tolerance
    print(f"Benign performance is: {benign_performance}, and tolerance is: {lower_bound_perf}")

    # Initialize masks for attention and MLP layers
    attn_masks = {}
    mlp_masks = {}
    for name, src_param_mlp in src_encoder.state_dict().items():
        if 'attn' in name and prune_attn:
            attn_masks[name] = torch.ones_like(src_param_mlp, device=src_param_mlp.device)
        elif 'mlp' in name and prune_mlp:
            mlp_masks[name] = torch.ones_like(src_param_mlp, device=src_param_mlp.device)

    # Perform pruning iteratively
    for t in range(1, iters + 1):
        if t > 1:
            perf_this_iter = eval_single_dataset(src_encoder, source_task, args)['top1']
            print(f"Performance after {t-1}-th pruning iteration with performance as: {perf_this_iter}")
            if perf_this_iter < lower_bound_perf:
                print(f"Stopping early at iteration {t-1}: Performance degraded too much.")
                if t == 2:
                    raise AttributeError("No pruning is performed for performance degradation!")
                src_encoder.load_state_dict(full_src_params_bkup)
                break
            else:
                full_src_params_bkup = {name: param.clone() for name, param in src_encoder.state_dict().items()}  # Back up params of last iteration
        print(f"Starting {t}-th pruning iteration.")

        for block_idx, (attn_ratio, mlp_ratio) in enumerate(zip(attn_pruning_ratios, mlp_pruning_ratios)):
            # Calculate per-iteration pruning ratio
            attn_ratio_per_iter = t * attn_ratio / iters
            mlp_ratio_per_iter = t * mlp_ratio / iters

            full_src_params = {name: param.clone() for name, param in src_encoder.state_dict().items()}

            # Filter attention parameters for the current block
            if prune_attn:
                block_prefix_attn = f"resblocks.{block_idx}.attn"
                block_params_attn = {name: param for name, param in src_encoder.state_dict().items() if block_prefix_attn in name}

                for name, src_param_attn in block_params_attn.items():
                    # Get the corresponding reference parameter
                    ref_param_attn = get_layer_by_name(reference_params, name).to(src_param_attn.device)

                    # Reset attn_masks for current iteration
                    attn_masks[name] = torch.ones_like(src_param_attn, device=src_param_attn.device)

                    # Compute importance scores based on the formula
                    importance_scores_attn = (alpha * torch.abs(src_param_attn) + beta * torch.abs(src_param_attn - ref_param_attn)) * attn_masks[name]

                    # Calculate the pruning threshold based on the per-iteration constraint
                    total_weights = importance_scores_attn.numel()
                    num_keep = int((1 - attn_ratio_per_iter) * total_weights)
                    if num_keep > 0:
                        threshold = torch.topk(importance_scores_attn.flatten(), num_keep, largest=True).values[-1]

                        # Update mask for the layer
                        attn_masks[name][importance_scores_attn < threshold] = 0

                        # Apply mask to the parameter
                        src_param_attn *= attn_masks[name]
                # Update attn params
                full_src_params.update(block_params_attn)

            if prune_mlp:
                # Filter MLP parameters for the current block
                block_prefix_mlp = f"resblocks.{block_idx}.mlp"
                block_params_mlp = {name: param for name, param in src_encoder.state_dict().items() if block_prefix_mlp in name}

                for name, src_param_mlp in block_params_mlp.items():
                    # Get the corresponding reference parameter
                    ref_param_mlp = get_layer_by_name(reference_params, name).to(src_param_mlp.device)

                    # Reset mlp_masks for current iteration
                    mlp_masks[name] = torch.ones_like(src_param_mlp, device=src_param_mlp.device)

                    # Compute importance scores based on the formula
                    importance_scores_mlp = (alpha * torch.abs(src_param_mlp) + beta * torch.abs(src_param_mlp - ref_param_mlp)) * mlp_masks[name]

                    # Calculate the pruning threshold based on the per-iteration constraint
                    total_weights = importance_scores_mlp.numel()
                    num_keep = int((1 - mlp_ratio_per_iter) * total_weights)
                    if num_keep > 0:
                        threshold = torch.topk(importance_scores_mlp.flatten(), num_keep, largest=True).values[-1]

                        # Update mask for the layer
                        mlp_masks[name][importance_scores_mlp < threshold] = 0

                        # Apply mask to the parameter
                        src_param_mlp *= mlp_masks[name]
                # Update MLP params
                full_src_params.update(block_params_mlp)
            # Update encoder
            src_encoder.load_state_dict(full_src_params)

        if t == iters:
            print(f"Finish {iters} round iteration of pruning in attn and mlp layers")

    perf_last_iter = eval_single_dataset(src_encoder, source_task, args)['top1']
    print(f"Performance after pruning iteration with performance as: {perf_last_iter}")
    if perf_last_iter < lower_bound_perf:
        print(f"Discraring last iteration: Performance degraded too much.")
        src_encoder.load_state_dict(full_src_params_bkup)
    return src_encoder



def get_layer_by_name(model, layer_name):
    # 将路径分割成列表
    parts = layer_name.split('.')
    layer = model
    for part in parts:
        if part.isdigit():
            layer = layer[int(part)]
        else:
            layer = getattr(layer, part)
    return layer


if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    # Config for addition
    source_task = 'DTD'
    free_rider_task = 'MNIST'
    model = 'ViT-B-32'
    vector_scaling_coef = 0.8
    args = parse_arguments()
    args.data_location = 'data'
    args.model = model
    args.save = f'checkpoints/{model}'
    args.results_db = f'results/Source_{source_task}_Dest_{free_rider_task}.txt'

    # Setting up pruning rule
    prune_attn = True
    prune_MLP = True
    prune_alpha = 2
    prune_beta = 1
    tolerance = 0.95

    attn_pruning_ratios = [0.8] * 12  # 12 layers!
    mlp_pruning_ratios = [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4]
    mlp_pruning_ratios.reverse()
    print(mlp_pruning_ratios)

    # Model paths
    pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
    source_task_checkpoint = f'checkpoints/{model}/{source_task}/finetuned.pt'
    source_task_checkpoint_permuted = (f'experiments/perm_all_layers/permuted models/black box/{model}/{source_task}/'
                                      f'victim_{source_task}_permuted.pt')
    free_rider_task_checkpoint = f'checkpoints/{model}/{free_rider_task}/finetuned.pt'

    # Load Model
    source_encoder = torch.load(source_task_checkpoint)
    source_permuted_encoder = torch.load(source_task_checkpoint_permuted)
    free_rider_encoder = torch.load(free_rider_task_checkpoint)

    results_dict = {'scaling coef': vector_scaling_coef}

    # # Evaluate benign performance
    # print(f"Evaluating Fine-tuned Source Performance on victim: {source_task}...")
    # victim_info = eval_single_dataset(source_encoder, source_task, args)
    # results_dict[source_task + '_benign'] = victim_info['top1']

    # Generate task vector
    T_src = TaskVector(pretrained_checkpoint, source_task_checkpoint)
    T_src_permuted = TaskVector(pretrained_checkpoint, source_task_checkpoint_permuted)
    T_free_rider = TaskVector(pretrained_checkpoint, free_rider_task_checkpoint)

    # Task arithmetic
    # combine_task_T = sum([T_src, T_free_rider])
    # combined_encoder = combine_task_T.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
    # combine_task_T_permuted = sum([T_src_permuted, T_free_rider])
    # combined_encoder_permuted = combine_task_T_permuted.apply_to(pretrained_checkpoint, scaling_coef=vector_scaling_coef)

    # # Evaluate permuted task arithmetic on victim -!
    # print(f"\nEvaluating Permuted TA... vic:{source_task}, fr:{free_rider_task} (Scaling coef: {vector_scaling_coef}):"
    #       f" on task: {source_task}...")
    # permuted_ta_info = eval_single_dataset(combined_encoder_permuted, source_task, args)
    # results_dict[source_task + '_TA_permuted'] = permuted_ta_info['top1']
    #
    # # Evaluate permuted task arithmetic on free-rider -!
    # print(f"\nEvaluating Permuted TA... vic:{source_task}, fr:{free_rider_task} (Scaling coef: {vector_scaling_coef}):"
    #       f" on task: {free_rider_task}...")
    # permuted_ta_fr_info = eval_single_dataset(combined_encoder_permuted, free_rider_task, args)
    # results_dict[free_rider_task + '_TA_permuted'] = permuted_ta_fr_info['top1']

    source_encoder_perm_prune = selective_prune_attention_and_mlp_layers(src_encoder=source_permuted_encoder,
                                                                         reference_params=free_rider_encoder,
                                                                         attn_pruning_ratios=attn_pruning_ratios,
                                                                         mlp_pruning_ratios=mlp_pruning_ratios,
                                                                         tolerance=tolerance,
                                                                         alpha=prune_alpha,
                                                                         beta=prune_beta,
                                                                         prune_attn=prune_attn,
                                                                         prune_mlp=prune_MLP)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f'experiments/data_free_pruning/perm_prune_models/{model}/{source_task}/{current_time}/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # model_name = f'victim_{source_task}_perm_prune_attn_and_mlp_a_{prune_alpha}_b_{prune_beta}_tol_{tolerance}.pt'
    model_name = f'victim_{source_task}_perm_prune_attn_and_mlp.pt'
    perm_prune_save_checkpoint = os.path.join(save_path, model_name)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    ratios_file = os.path.join(save_path, "pruning_params.txt")
    with open(ratios_file, "w") as f:
        f.write("Attention Pruning Ratios: " + ", ".join(map(str, attn_pruning_ratios)) + "\n")
        f.write("MLP Pruning Ratios: " + ", ".join(map(str, mlp_pruning_ratios)) + "\n")
        f.write(f"Alpha: {prune_alpha}\n")
        f.write(f"Beta: {prune_beta}\n")
        f.write(f"Tolerance: {tolerance}\n")

    source_encoder_perm_prune.to('cpu')
    torch.save(source_encoder_perm_prune, perm_prune_save_checkpoint)

    # Get task-vectors
    T_src_perm_prune = TaskVector(pretrained_checkpoint, perm_prune_save_checkpoint)
    combine_perm_prune_T = sum([T_src_perm_prune, T_free_rider])
    combined_perm_prune_encoder = combine_perm_prune_T.apply_to(pretrained_checkpoint, scaling_coef=vector_scaling_coef)

    # Evaluate benign performance on victim -!
    print(f"\nEvaluating benign performance after pruning on task: {source_task}:")
    be_info = eval_single_dataset(source_encoder_perm_prune, source_task, args)
    results_dict[source_task + '_perm_prune_benign'] = be_info['top1']

    # Evaluate perm_prune task arithmetic on victim -!
    print(f"\nEvaluating perm_prune TA... vic:{source_task}, fr:{free_rider_task} (Scaling coef: {vector_scaling_coef}):"
          f" on task: {source_task}...")
    perm_prune_ta_info = eval_single_dataset(combined_perm_prune_encoder, source_task, args)
    results_dict[source_task + '_perm_prune_TA'] = perm_prune_ta_info['top1']

    # Evaluate perm_prune task arithmetic on free-rider -!
    print(f"\nEvaluating perm_prune TA... vic:{source_task}, fr:{free_rider_task} (Scaling coef: {vector_scaling_coef}):"
          f" on task: {free_rider_task}...")
    perm_prune_ta_fr_info = eval_single_dataset(combined_perm_prune_encoder, free_rider_task, args)
    results_dict[free_rider_task + '_perm_prune_TA'] = perm_prune_ta_fr_info['top1']

    for k, v in results_dict.items():
        print(k, v)


"""
算法可调参数：权重比例 alpha & beta, 剪枝比例 pruning_ratio, 容忍度 tolerance


1. 仅prune attn还是可逆，这也太逆天了 （见prune_attention_layers执行结果）；

Experiment data:
    BENIGN_DTD_MNIST_TA: 0.8521 (sc 0.8); 0.6951 (sc 0.3)
    
    1. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.3, 0.4, 0.5] (alpha=beta=1) - 实际剪枝2/4
        scaling coef 0.8
        DTD_perm_prune_benign 0.8938
        DTD_perm_prune_TA 0.0399
        MNIST_perm_prune_TA 0.0958
        DTD_reversed_TA: 0.7635
        MNIST_reversed_TA: 0.9959

    2. VIC DTD (MLP) - [0.1, 0.1, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.3, 0.4, 0.5] (alpha=beta=1) - 实际剪枝2/4
        scaling coef 0.8
        DTD_perm_prune_benign 0.8920
        DTD_perm_prune_TA 0.0356
        MNIST_perm_prune_TA 0.0958
        DTD_reversed_TA: 0.7661
        MNIST_reversed_TA: 0.9960
        
    --------------从1，2看来，多剪一个attn没什么变化。接下来比较同pruning下的alpha和beta-----------------------------------------------

    3. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4] (alpha=1, beta=2) - 实际剪枝2/5
        scaling coef 0.8
        DTD_perm_prune_benign 0.8790
        DTD_perm_prune_TA 0.0409
        MNIST_perm_prune_TA 0.0950
        DTD_reversed_TA: 0.7416
        MNIST_reversed_TA: 0.9960
    
    
    4. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4] (alpha=2, beta=1) - 实际剪枝3/5
        DTD_perm_prune_benign 0.8875
        DTD_perm_prune_TA 0.0445
        MNIST_perm_prune_TA 0.0888
        DTD_reversed_TA: 0.7561
        MNIST_reversed_TA: 0.9963
        
        sc 0.3
        "DTD_reversed_TA": 0.6165
        "MNIST_reversed_TA": 0.9697
        
    5. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4] (alpha=1, beta=0) - 实际剪枝5/5
        scaling coef 0.8
        DTD_perm_prune_benign 0.8915
            DTD_perm_prune_TA 0.0408
        MNIST_perm_prune_TA 0.0813
        DTD_reversed_TA: 0.8044
        MNIST_reversed_TA: 0.9965

    6. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4] (alpha=0, beta=1) - 实际剪枝0/5
    
        一剪性能就炸了！

    7. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4] (alpha=3, beta=1) - 实际剪枝4/5
        scaling coef 0.8
        DTD_perm_prune_benign 0.8740
        DTD_perm_prune_TA 0.0460
        MNIST_perm_prune_TA 0.0919
        DTD_reversed_TA: 0.7368
        MNIST_reversed_TA: 0.9962
        
        sc 0.3
        "DTD_reversed_TA": 0.6140
        "MNIST_reversed_TA": 0.9700
        
    8. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4] (alpha=4, beta=1) - 实际剪枝4/5
        scaling coef 0.8
        DTD_perm_prune_benign 0.8843
        DTD_perm_prune_TA 0.0438
        MNIST_perm_prune_TA 0.0873
        DTD_reversed_TA: 0.7670
        MNIST_reversed_TA: 0.9963
        
        sc 0.3
        "DTD_reversed_TA": 0.6177
        "MNIST_reversed_TA": 0.9683
        
    9. VIC DTD (MLP ATTN) - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4] (alpha=5, beta=1) - 实际剪枝4/5
        scaling coef 0.8
        DTD_perm_prune_benign 0.8716
        DTD_perm_prune_TA 0.0361
        MNIST_perm_prune_TA 0.0960
        DTD_reversed_TA: 0.7536
        MNIST_reversed_TA: 0.9964
        
        sc 0.3 
        "DTD_reversed_TA": 0.6133
        "MNIST_reversed_TA": 0.9672
   
    10. VIC DTD (MLP ATTN) - [0.3, 0.25, 0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.08, 0.05, 0.05, 0.05] (alpha=2, beta=1) 
        scaling coef 0.8
        DTD_perm_prune_benign 0.9054
        DTD_perm_prune_TA 0.0379
        MNIST_perm_prune_TA 0.0646
        
    11. VIC DTD (Non perm! MLP ATTN)  - [0.1, 0.1, 0.15, 0.15, 0.15, 0.2, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4](alpha=2, beta=1) 
        scaling coef 0.8
        DTD_perm_prune_benign 0.8895
        DTD_perm_prune_TA 0.8210
        MNIST_perm_prune_TA 0.9967
        DTD_reversed_TA: 0.8210
        MNIST_reversed_TA: 0.9967

    13. VIC DTD ( MLP ATTN)  check 11-19-28 
        scaling coef 0.8
        DTD_perm_prune_benign 0.8950473806543894
        DTD_perm_prune_TA 0.04201680672268908
        MNIST_perm_prune_TA 0.0692

"""

