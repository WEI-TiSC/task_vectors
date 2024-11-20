import os
import matplotlib.pyplot as plt


task = 'MNIST'
free_rider = 'SVHN'
coef = 0.8

layers = [[i, i+1, i+2] for i in range(0, 10)]
layer_labels = [f"[{layer[0]},{layer[1]},{layer[2]}]" for layer in layers]
benign_ta = [0.9938] * 10
benign_ta_other = [0.9227] * 10
perm_vi = [0.7764, 0.9788, 0.9775, 0.9395, 0.9651, 0.9869, 0.9923, 0.9935, 0.9941, 0.9939]
perm_fr = [0.4314, 0.5090, 0.6298, 0.6303, 0.6839, 0.7994, 0.8863, 0.9077, 0.9169, 0.9190]

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
            f'Vic_{task}_Fr_{free_rider}_coef_{coef}_plot_perm_3_layers.png'), dpi=300)

# Display plot
plt.show()