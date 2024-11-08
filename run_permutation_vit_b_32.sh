# EuroSAT
python matching_weights.py --victim_task EuroSAT --free_rider_task GTSRB --base_model ViT-B-32 --scaling_coef 0.3
python matching_weights.py --victim_task EuroSAT --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.3

# GTSRB
python matching_weights.py --victim_task GTSRB --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.3
python matching_weights.py --victim_task GTSRB --free_rider_task MNIST --base_model ViT-B-32 --scaling_coef 0.3
python matching_weights.py --victim_task GTSRB --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.3
python matching_weights.py --victim_task GTSRB --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.3