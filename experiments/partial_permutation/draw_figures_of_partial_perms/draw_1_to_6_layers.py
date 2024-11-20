import os
import matplotlib.pyplot as plt


task = 'MNIST'
free_rider = 'SVHN'
coef = 0.8

layers = ['[0]', '[0,1]', '[0,1,2]', '[0,1,2,3]', '[0,1,2,3,4]', '[0,1,2,3,4,5]']
benign_ta = [0.9938] * 6
benign_ta_other = [0.9227] * 6
perm_vi = [0.9889, 0.7264, 0.7764, 0.3196, 0.1466, 0.1657]
perm_fr = [0.9075, 0.4441, 0.4314, 0.2178, 0.1995, 0.1599]

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
            f'Vic_{task}_Fr_{free_rider}_coef_{coef}_plot_perm_1_to_6_layers.png'), dpi=300)

# Display plot
plt.show()