"""transforms.py  (OWNER: Member P). Stand-in copy."""
import numpy as np
from scipy.stats import rankdata

TRANSFORMS = {"identity": lambda y: y, "log1p": np.log1p, "arcsinh": np.arcsinh, "sqrt": np.sqrt}
SCALES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def apply_transform(name, y, a=1.0):
    if name == "rank":  # unit-invariant control
        return rankdata(y) / len(y)
    return TRANSFORMS[name](a * y)
