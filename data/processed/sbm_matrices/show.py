import numpy as np
from pathlib import Path

here = Path(__file__).parent
np.set_printoptions(precision=10, suppress=False, linewidth=120)

k = int(np.load(here / "k.npy")[0])
sizes = np.load(here / "class_sizes.npy")
bp = np.load(here / "b_plus.npy")
bm = np.load(here / "b_minus.npy")

print(f"k = {k}")
print(f"class_sizes = {sizes.tolist()}  (total = {sizes.sum()})")
print()
print("b_plus:")
print(bp)
print()
print("b_minus:")
print(bm)
print()
print(f"b_plus  diagonal : {np.diag(bp)}")
print(f"b_minus diagonal : {np.diag(bm)}")
print()

mask = ~np.eye(k, dtype=bool)
bp_off = bp[mask]
bm_off = bm[mask]
print(f"b_plus  off-diagonal range : [{bp_off.min():.3e}, {bp_off.max():.3e}]")
print(f"b_minus off-diagonal range : [{bm_off.min():.3e}, {bm_off.max():.3e}]")
print()
ratio = bm_off / np.where(bp_off > 0, bp_off, np.nan)
print(f"b_minus/b_plus off-diagonal ratios (expect ~522x if correct):")
r_mat = np.full((k, k), np.nan)
r_mat[mask] = ratio
print(np.round(r_mat, 1))
print()
print(f"b_plus == b_minus : {np.array_equal(bp, bm)}")
print(f"max abs diff      : {np.abs(bp - bm).max():.3e}")
