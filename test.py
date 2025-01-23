import torch

from src.task_vectors import TaskVector
from src.eval import eval_single_dataset
from src.args import parse_arguments


# Config for addition
source_task = 'MNIST'
dest_task = 'Cars'
model = 'ViT-B-32'
vector_scaling_coef = 0.4
args = parse_arguments()
args.data_location = 'data'
args.model = model
args.save = f'checkpoints/{model}'
args.results_db = f'results/Source_{source_task}_Dest_{dest_task}.txt'
# pretrained_checkpoint = f'checkpoints/{model}/zeroshot.pt'
source_task_checkpoint = f'checkpoints/{model}/{source_task}/finetuned.pt'
# dest_task_checkpoint = f'checkpoints/{model}/{dest_task}/finetuned.pt'
# victm_task_checkpoint_perm_prune = (f'experiments/data_free_pruning/perm_prune_models/{model}/'
#                                     f'victim_{source_task}_perm_prune.pt')


if __name__ == "__main__":
    container = []
    source_image_encoder = torch.load(source_task_checkpoint)
    # dest_image_encoder = torch.load(dest_task_checkpoint)
    src_params = {name: param.clone() for name, param in source_image_encoder.state_dict().items()}
                  # if 'attn' in name or 'mlp' in name}
    for eachk, v in src_params.items():
        print(eachk)

    # print(f"Evaluating Fine-tuned Source Performance on {source_task}...")
    # source_info = eval_single_dataset(source_image_encoder, source_task, args)
    # container.append(source_info)
