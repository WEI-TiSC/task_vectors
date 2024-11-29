import sys
import os
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
os.chdir(root_dir)
os.environ["http_proxy"] = "http://proxy.noc.titech.ac.jp:3128/"
os.environ["https_proxy"] = 'http://proxy.noc.titech.ac.jp:3128/'

import time

from src.args import parse_arguments
from src.finetune import finetune


if __name__ == "__main__":
    start_time = time.time()
    data_location = 'data'
    # models = ['ViT-B-32', 'ViT-B-16', 'ViT-L-14']
    models = ['ViT-B-32']
    # datasets = ['Cars', 'DTD', 'EuroSAT', 'GTSRB', 'MNIST', 'RESISC45', 'SUN397', 'SVHN']
    datasets = ['MNIST']
    epochs = {
        'Cars': 35,
        'DTD': 76,
        'EuroSAT': 12,
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SUN397': 14,
        'SVHN': 4,
        'ImageNet': 4
    }

    for model in models:
        for dataset in datasets:
            print('=' * 100)
            print(f'Finetuning {model} on {dataset}')
            print('=' * 100)
            args = parse_arguments()
            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.train_dataset = dataset
            args.batch_size = 128
            args.model = model
            args.save = f'checkpoints/{model}'
            finetune(args)
    print(f"Time consume: {time.time()-start_time}")