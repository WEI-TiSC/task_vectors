import os
import matplotlib.pyplot as plt


task = 'MNIST'
free_rider = 'SVHN'
coef = 0.8

layers = [[i, i+1] for i in range(0, 11)]
layer_labels = [f"[{layer[0]},{layer[1]}]" for layer in layers]
benign_ta = [0.9938] * 11
benign_ta_other = [0.9227] * 11
perm_vi = [0.7264, 0.9889, 0.9899, 0.9831, 0.9746, 0.9886, 0.9925, 0.9935, 0.9937, 0.9942, 0.9942]
perm_fr = [0.4441, 0.7430, 0.7677, 0.7983, 0.7481, 0.8184, 0.8882, 0.9110, 0.9224, 0.9197, 0.9226]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(layer_labels, benign_ta, label='Benign TA victim', marker='o')
plt.plot(layer_labels, benign_ta_other, label='Benign TA free-rider', marker='s')
plt.plot(layer_labels, perm_vi, label='Permuted victim', marker='^')
plt.plot(layer_labels, perm_fr, label='Permuted free-rider', marker='x')

# Adding labels, legend, and grid
plt.xlabel('Layers')
plt.ylabel('Values')
plt.title('Line Plot of Benign TA victim, free-rider, Permuted victim and free-rider')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'results',
            f'Vic_{task}_Fr_{free_rider}_coef_{coef}_plot_perm_2_layers.png'), dpi=300)

# Display plot
plt.show()
