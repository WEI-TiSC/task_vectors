# Interpretation: Try orthogonal transformation.
import torch
import torch.nn.functional as F

from src.args import parse_arguments
from src.task_vectors import TaskVector
from src.eval import eval_single_dataset

# Configs
source_task = 'GTSRB'
dest_task = 'MNIST'
model = 'ViT-B-32'
vector_scaling_coef = 1.0

# Orthogonal trans layers
layers = [i for i in range(12)]

args = parse_arguments()
args.data_location = 'data'
args.model = model
args.save = f'checkpoints/{model}'

source_task_checkpoint = f'checkpoints/{model}/{source_task}/finetuned.pt'  # Vector to be merged...
pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'  # Pre-trained checkpoint for T_source
dest_task_checkpoint = f'checkpoints/{model}/{dest_task}/finetuned.pt'  # Theta_dest, who wants T_source

image_encoder = torch.load(source_task_checkpoint)


def orthononal_in_mlp(layer: int):
    # Get Resblock
    resblock = image_encoder.model.visual.transformer.resblocks[layer]

    # Extract FC layers
    W_fc = resblock.mlp.c_fc.weight.data.clone()
    W_proj = resblock.mlp.c_proj.weight.data.clone()

    # Generate orthogonal matrix
    Q_fc, R_fc = torch.linalg.qr(W_fc, 'complete')  # [fc_dim_out, fc_dim_out]
    Q_proj, R_proj = torch.linalg.qr(W_proj.T, 'complete')  # [proj_dim_in, proj_dim_in]

    # Change weights
    resblock.mlp.c_fc.weight.data = Q_fc @ W_fc
    resblock.mlp.c_proj.weight.data = W_proj @ Q_proj


if __name__ == "__main__":
    print(f"Evaluating Fine-tuned Source Performance on {source_task}...")
    source_info = eval_single_dataset(image_encoder, source_task, args)

    T_source = TaskVector(pretrained_checkpoint, source_task_checkpoint)  # T_source = theta_source - theta_pre
    print(f"Evaluating {dest_task}_prime after getting task vector of {source_task} (Scaling coef:"
          f" {vector_scaling_coef}): Performace on task: {source_task}...")
    editted_image_encoder = T_source.apply_to(dest_task_checkpoint, scaling_coef=vector_scaling_coef)  # Theta_dest' = Theta_dest + T_source
    editted_dest_info = eval_single_dataset(editted_image_encoder, source_task, args)
    #
    # image_encoder.to('cpu')
    #
    # # Orthogonal Transmission Setting Up
    # print(f"Start Orthogonal transmission in Resblocks: {layers}")
    # for layer in layers:
    #     orthononal_in_mlp(layer)
    #
    # torch.save(image_encoder, 'Orth_trans.pt')
    # orth_trans_checkpoint = 'Orth_trans.pt'
    #
    # print(f"Evaluating orth trans vector on Source Performance on {source_task}...")
    # source_info = eval_single_dataset(image_encoder, source_task, args)
    #
    # T_orth =TaskVector(pretrained_checkpoint, orth_trans_checkpoint)
    # print(f"Evaluating {dest_task}_prime with T_orth vector of {source_task} (Scaling coef:"
    #       f" {vector_scaling_coef}): Performace on task: {source_task}...")
    # orth_trans_image_encoder = T_orth.apply_to(dest_task_checkpoint, scaling_coef=vector_scaling_coef)
    # flipped_editted_dest_info = eval_single_dataset(orth_trans_image_encoder, source_task, args)
