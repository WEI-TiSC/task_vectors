# Grey box. Known one possible dataset(MNIST) . For MNIST model, known dataset is DTD.
# MNIST
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint SVHN
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint SVHN
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint SVHN
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint SVHN
python check_grey_and_black_box.py --victim_task MNIST --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint SVHN

# DTD
python check_grey_and_black_box.py --victim_task DTD --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task DTD --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task DTD --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task DTD --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task DTD --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint MNIST

# SVHN
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task SVHN --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint MNIST

# GTSRB
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task GTSRB --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint MNIST

# EuroSAT
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task EuroSAT --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint MNIST

# Cars
python check_grey_and_black_box.py --victim_task Cars --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task Cars --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task Cars --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task Cars --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task Cars --free_rider_task RESISC45 --base_model ViT-B-32 --perm_checkpoint MNIST

# RESISC45
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task SVHN --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task GTSRB --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task Cars --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task EuroSAT --base_model ViT-B-32 --perm_checkpoint MNIST
python check_grey_and_black_box.py --victim_task RESISC45 --free_rider_task DTD --base_model ViT-B-32 --perm_checkpoint MNIST