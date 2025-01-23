import os
import matplotlib.pyplot as plt


task = 'MNIST'
free_rider = 'SVHN'
coef = 0.8

layers = [[i, i+1, i+2, i+3, i+4, i+5] for i in range(0, 7)]
layer_labels = [f"[{layer[0]},{layer[1]},{layer[2]}, {layer[3]}, {layer[4]}, {layer[5]}]" for layer in layers]
benign_ta = [0.9938] * 7
benign_ta_other = [0.9227] * 7
perm_vi = [0.1659, 0.4832, 0.7169, 0.8660, 0.9463, 0.9846, 0.9914]
perm_fr = [0.1597, 0.1779, 0.3129, 0.5041, 0.6286, 0.7425, 0.8615]

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
            f'Vic_{task}_Fr_{free_rider}_coef_{coef}_plot_perm_6_layers.png'), dpi=300)

# Display plot
plt.show()