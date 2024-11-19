# Black box. Known pretrain checkpoint only
# MNIST
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint pretrain

# DTD
python check_grey_and_black_box.py --victim_task DTD --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task DTD --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task DTD --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task DTD --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task DTD --free_rider_task MNIST --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task DTD --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint pretrain

# SVHN
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task MNIST --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint pretrain

# GTSRB
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task MNIST --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint pretrain

# EuroSAT
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task MNIST --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint pretrain

# Cars
python check_grey_and_black_box.py --victim_task Cars --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task Cars --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task Cars --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task Cars --free_rider_task MNIST --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task Cars --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task Cars --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint pretrain

# RESISC45
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task MNIST --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint pretrain
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint pretrain