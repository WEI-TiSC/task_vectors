# Run with benign/ permuted performance
#python check_model_attributes.py --victim_task EuroSAT --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task MNIST --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SVHN --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task GTSRB --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task DTD --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task Cars --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SUN397 --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task RESISC45 --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.8


# Run without benign/ permuted performance

# -----------------------------------------------------EuroSAT-----------------------------------------------------------
python check_model_attributes.py --victim_task EuroSAT --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task EuroSAT --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.3

#python check_model_attributes.py --victim_task EuroSAT --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task EuroSAT --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task EuroSAT --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task EuroSAT --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.3

# -----------------------------------------------------MNIST-----------------------------------------------------------
python check_model_attributes.py --victim_task MNIST --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task MNIST --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.3

#python check_model_attributes.py --victim_task MNIST --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task MNIST --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task MNIST --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task MNIST --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.3

# -----------------------------------------------------SVHN-----------------------------------------------------------

python check_model_attributes.py --victim_task SVHN --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task SVHN --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.3

#python check_model_attributes.py --victim_task SVHN --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SVHN --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task SVHN --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task SVHN --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.3

# -----------------------------------------------------GTSRB-----------------------------------------------------------

python check_model_attributes.py --victim_task GTSRB --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task GTSRB --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.3

#python check_model_attributes.py --victim_task GTSRB --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task GTSRB --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task GTSRB --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task GTSRB --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.3

# -----------------------------------------------------DTD-----------------------------------------------------------

python check_model_attributes.py --victim_task DTD --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task DTD --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.3

#python check_model_attributes.py --victim_task DTD --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task DTD --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task DTD --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task DTD --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.3

# -----------------------------------------------------Cars-----------------------------------------------------------
python check_model_attributes.py --victim_task Cars --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task Cars --free_rider_task MNIST --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task Cars --free_rider_task MNIST --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task Cars --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task Cars --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task Cars --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task Cars --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task Cars --free_rider_task GTSRB --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task Cars --free_rider_task GTSRB --base_model ViT-B-32 --scaling_coef 0.3

#python check_model_attributes.py --victim_task Cars --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task Cars --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task Cars --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task Cars --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.3

## -----------------------------------------------------sun397-----------------------------------------------------------
#python check_model_attributes.py --victim_task SUN397 --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.3
#
#python check_model_attributes.py --victim_task SUN397 --free_rider_task MNIST --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SUN397 --free_rider_task MNIST --base_model ViT-B-32 --scaling_coef 0.3
#
#python check_model_attributes.py --victim_task SUN397 --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SUN397 --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.3
#
#python check_model_attributes.py --victim_task SUN397 --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SUN397 --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.3
#
#python check_model_attributes.py --victim_task SUN397 --free_rider_task GTSRB --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SUN397 --free_rider_task GTSRB --base_model ViT-B-32 --scaling_coef 0.3
#
#python check_model_attributes.py --victim_task SUN397 --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SUN397 --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.3
#
#python check_model_attributes.py --victim_task SUN397 --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task SUN397 --free_rider_task RESISC45 --base_model ViT-B-32 --scaling_coef 0.3

# -----------------------------------------------------RESISC45-----------------------------------------------------------
python check_model_attributes.py --victim_task RESISC45 --free_rider_task DTD --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task RESISC45 --free_rider_task MNIST --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task RESISC45 --free_rider_task MNIST --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task RESISC45 --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task RESISC45 --free_rider_task SVHN --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task RESISC45 --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task RESISC45 --free_rider_task EuroSAT --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task RESISC45 --free_rider_task GTSRB --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task RESISC45 --free_rider_task GTSRB --base_model ViT-B-32 --scaling_coef 0.3

python check_model_attributes.py --victim_task RESISC45 --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.8
python check_model_attributes.py --victim_task RESISC45 --free_rider_task Cars --base_model ViT-B-32 --scaling_coef 0.3

#python check_model_attributes.py --victim_task RESISC45 --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.8
#python check_model_attributes.py --victim_task RESISC45 --free_rider_task SUN397 --base_model ViT-B-32 --scaling_coef 0.3