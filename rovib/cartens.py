import numpy as np

# Transformation matrix from Cartesian to spherical-tensor representation for rank-1 tensor

_UMAT_RANK1 = {
    1: np.array(
        [
            [np.sqrt(2) / 2, -np.sqrt(2) * 1j / 2, 0],
            [0, 0, 1],
            [-np.sqrt(2) / 2, -np.sqrt(2) * 1j / 2, 0],
        ],
        dtype=np.complex128,
    ),
}

# Transformation matrix from Cartesian to spherical-tensor representation for rank-2 tensor

_UMAT_RANK2 = {
    0: np.array(
        [[-1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3)]]
    ),
    1: np.array(
        [
            [0, 0, -0.5, 0, 0, 0.5 * 1j, 0.5, -0.5 * 1j, 0],
            [0, 1.0 / np.sqrt(2) * 1j, 0, -1.0 / np.sqrt(2) * 1j, 0, 0, 0, 0, 0],
            [0, 0, -0.5, 0, 0, -0.5 * 1j, 0.5, 0.5 * 1j, 0],
        ],
        dtype=np.complex128,
    ),
    2: np.array(
        [
            [0.5, -0.5 * 1j, 0, -0.5 * 1j, -0.5, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, -0.5 * 1j, 0.5, -0.5 * 1j, 0],
            [-1 / np.sqrt(6), 0, 0, 0, -1 / np.sqrt(6), 0, 0, 0, (1 / 3) * np.sqrt(6)],
            [0, 0, -0.5, 0, 0, -0.5 * 1j, -0.5, -0.5 * 1j, 0],
            [0.5, 0.5 * 1j, 0, 0.5 * 1j, -0.5, 0, 0, 0, 0],
        ],
        dtype=np.complex128,
    ),
}

UMAT_CART_TO_SPHER = {1: _UMAT_RANK1, 2: _UMAT_RANK2}

UMAT_SPHER_TO_CART = {
    rank: np.linalg.pinv(np.concatenate(list(tens.values()), axis=0))
    for rank, tens in UMAT_CART_TO_SPHER.items()
}

# Cartesian components and irreducible representations for tensors of different ranks

CART_IND = {
    1: ["x", "y", "z"],
    2: ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"],
}

SPHER_IND = {
    1: [(1, -1), (1, 0), (1, 1)],
    2: [(o, s) for o in range(3) for s in range(-o, o + 1)],
}