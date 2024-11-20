import os
import matplotlib.pyplot as plt


task = 'MNIST'
free_rider = 'SVHN'
coef = 0.8

layers = [i for i in range(12)]
benign_ta = [0.9938] * 12
benign_ta_other = [0.9227] * 12
perm_vi = [0.9889, 0.9925, 0.9926, 0.9927, 0.9909, 0.9911, 0.9927, 0.9936, 0.9938, 0.9941, 0.9941, 0.9938]
perm_fr = [0.9075, 0.9053, 0.8813, 0.8756, 0.8661, 0.8600, 0.8998, 0.9166, 0.9214, 0.9216, 0.9237, 0.9217]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(layers, benign_ta, label='Benign TA victim', marker='o')
plt.plot(layers, benign_ta_other, label='Benign TA free-rider', marker='s')
plt.plot(layers, perm_vi, label='Permuted victim', marker='^')
plt.plot(layers, perm_fr, label='Permuted free-rider', marker='x')

# Adding labels, legend, and grid
plt.xlabel('Layers')
plt.ylabel('Values')
plt.title('Line Plot of Benign TA victim, free-rider, Permuted victim and free-rider')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), 'results',
                         f'Vic_{task}_Fr_{free_rider}_coef_{coef}_plot_perm_1_layer.png'), dpi=300)

# Display plot
plt.show()