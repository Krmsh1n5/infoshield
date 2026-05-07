import numpy as np
b_plus  = np.load('data/processed/sbm_matrices/b_plus.npy')
b_minus = np.load('data/processed/sbm_matrices/b_minus.npy')
k = b_plus.shape[0]
print(f"k = {k}")
mask = np.eye(k, dtype=bool)
print(f"b+ diagonal/off-diagonal ratio: {b_plus[mask].mean() / b_plus[~mask].mean():.1f}x")
print(f"b- off-diagonal/diagonal ratio: {b_minus[~mask].mean() / b_minus[mask].mean():.1f}x")
# Want both ratios as large as possible for strong discrimination