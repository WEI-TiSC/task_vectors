# MNIST
python matching_weights.py --victim_task MNIST --free_rider_task Cars --base_model ViT-B-32
#python matching_weights.py --victim_task MNIST --free_rider_task SUN397 --base_model ViT-B-32
python matching_weights.py --victim_task MNIST --free_rider_task RESISC45 --base_model ViT-B-32

# EuroSAT
python matching_weights.py --victim_task EuroSAT --free_rider_task Cars --base_model ViT-B-32
#python matching_weights.py --victim_task EuroSAT --free_rider_task SUN397 --base_model ViT-B-32
python matching_weights.py --victim_task EuroSAT --free_rider_task RESISC45 --base_model ViT-B-32

# GTSRB
python matching_weights.py --victim_task GTSRB --free_rider_task Cars --base_model ViT-B-32
#python matching_weights.py --victim_task GTSRB --free_rider_task SUN397 --base_model ViT-B-32
python matching_weights.py --victim_task GTSRB --free_rider_task RESISC45 --base_model ViT-B-32

# DTD
python matching_weights.py --victim_task DTD --free_rider_task Cars --base_model ViT-B-32
#python matching_weights.py --victim_task DTD --free_rider_task SUN397 --base_model ViT-B-32
python matching_weights.py --victim_task DTD --free_rider_task RESISC45 --base_model ViT-B-32

# SVHN
python matching_weights.py --victim_task SVHN --free_rider_task Cars --base_model ViT-B-32
#python matching_weights.py --victim_task SVHN --free_rider_task SUN397 --base_model ViT-B-32
python matching_weights.py --victim_task SVHN --free_rider_task RESISC45 --base_model ViT-B-32

# --------------------------------------------------------------------------------------------------------------------
# RESISC45
python matching_weights.py --victim_task RESISC45 --free_rider_task MNIST --base_model ViT-B-32
python matching_weights.py --victim_task RESISC45 --free_rider_task SVHN --base_model ViT-B-32
python matching_weights.py --victim_task RESISC45 --free_rider_task DTD --base_model ViT-B-32
python matching_weights.py --victim_task RESISC45 --free_rider_task EuroSAT --base_model ViT-B-32
python matching_weights.py --victim_task RESISC45 --free_rider_task Cars --base_model ViT-B-32
#python matching_weights.py --victim_task RESISC45 --free_rider_task SUN397 --base_model ViT-B-32
python matching_weights.py --victim_task RESISC45 --free_rider_task GTSRB --base_model ViT-B-32

## sun397
#python matching_weights.py --victim_task SUN397 --free_rider_task MNIST --base_model ViT-B-32
#python matching_weights.py --victim_task SUN397 --free_rider_task SVHN --base_model ViT-B-32
#python matching_weights.py --victim_task SUN397 --free_rider_task DTD --base_model ViT-B-32
#python matching_weights.py --victim_task SUN397 --free_rider_task EuroSAT --base_model ViT-B-32
#python matching_weights.py --victim_task SUN397 --free_rider_task Cars --base_model ViT-B-32
#python matching_weights.py --victim_task SUN397 --free_rider_task RESISC45 --base_model ViT-B-32
#python matching_weights.py --victim_task SUN397 --free_rider_task GTSRB --base_model ViT-B-32

# Cars
python matching_weights.py --victim_task Cars --free_rider_task MNIST --base_model ViT-B-32
python matching_weights.py --victim_task Cars --free_rider_task SVHN --base_model ViT-B-32
python matching_weights.py --victim_task Cars --free_rider_task DTD --base_model ViT-B-32
python matching_weights.py --victim_task Cars --free_rider_task EuroSAT --base_model ViT-B-32
#python matching_weights.py --victim_task Cars --free_rider_task SUN397 --base_model ViT-B-32
python matching_weights.py --victim_task Cars --free_rider_task RESISC45 --base_model ViT-B-32
python matching_weights.py --victim_task Cars --free_rider_task GTSRB --base_model ViT-B-32