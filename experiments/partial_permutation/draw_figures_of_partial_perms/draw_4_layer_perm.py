import os
import matplotlib.pyplot as plt


task = 'MNIST'
free_rider = 'SVHN'
coef = 0.8

layers = [[i, i+1, i+2, i+3] for i in range(0, 9)]
layer_labels = [f"[{layer[0]},{layer[1]},{layer[2]}, {layer[3]}]" for layer in layers]
benign_ta = [0.9938] * 9
benign_ta_other = [0.9227] * 9
perm_vi = [0.3196, 0.9427, 0.9030, 0.8983, 0.9568, 0.9864, 0.9921, 0.9941, 0.9941]
perm_fr = [0.2178, 0.3576, 0.4152, 0.5060, 0.6568, 0.7873, 0.8810, 0.8999, 0.9139]

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
            f'Vic_{task}_Fr_{free_rider}_coef_{coef}_plot_perm_4_layers.png'), dpi=300)

# Display plot
plt.show()