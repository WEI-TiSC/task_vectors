# Try MNIST

# 1 layer
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [0]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [1]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [2]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [3]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [4]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [5]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [6]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [7]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [8]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [9]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [10]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [11]

# 2 layers
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [0,1]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [2,3]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [4,5]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [6,7]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [8,9]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [10,11]

# 3 layers
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [0,1,2]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [3,4,5]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [6,7,8]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [9,10,11]

# 4 layers
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [0,1,2,3]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [4,5,6,7]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [8,9,10,11]

# 5 layers
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [0,1,2,3,4]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [4,5,6,7,8]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [7,8,9,10,11]

# 6 layers
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [0,1,2,3,4,5]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [4,5,6,7,8,9]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [6,7,8,9,10,11]

#random
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [1,3,5,7,9]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [2,4,6,8,10]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [1,3,5,7,9,11]
python experiments/partial_permutation/partial_perms_black_box.py --victim_task MNIST --base_model ViT-B-32 --perm_layers [0,2,4,6,8,10]


