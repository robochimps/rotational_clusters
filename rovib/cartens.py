import numpy as np

# Transformation matrix from Cartesian to spherical-tensor representation
# for tensors of different ranks (dict keys)

UMAT_CART_TO_SPHER = {
    1: np.array(
        [
            [np.sqrt(2.0) / 2.0, -np.sqrt(2.0) * 1j / 2.0, 0],
            [0, 0, 1.0],
            [-np.sqrt(2.0) / 2.0, -np.sqrt(2.0) * 1j / 2.0, 0],
        ],
        dtype=np.complex128,
    ),
    2: np.array(
        [
            [
                -1.0 / np.sqrt(3.0),
                0,
                0,
                0,
                -1.0 / np.sqrt(3.0),
                0,
                0,
                0,
                -1.0 / np.sqrt(3.0),
            ],
            [0, 0, -0.5, 0, 0, 0.5 * 1j, 0.5, -0.5 * 1j, 0],
            [
                0,
                1.0 / np.sqrt(2.0) * 1j,
                0,
                -1.0 / np.sqrt(2.0) * 1j,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, -0.5, 0, 0, -0.5 * 1j, 0.5, 0.5 * 1j, 0],
            [0.5, -0.5 * 1j, 0, -0.5 * 1j, -0.5, 0, 0, 0, 0],
            [0, 0, 0.5, 0, 0, -0.5 * 1j, 0.5, -0.5 * 1j, 0],
            [
                -1.0 / np.sqrt(6.0),
                0,
                0,
                0,
                -1.0 / np.sqrt(6.0),
                0,
                0,
                0,
                (1.0 / 3.0) * np.sqrt(6.0),
            ],
            [0, 0, -0.5, 0, 0, -0.5 * 1j, -0.5, -0.5 * 1j, 0],
            [0.5, 0.5 * 1j, 0, 0.5 * 1j, -0.5, 0, 0, 0, 0],
        ],
        dtype=np.complex128,
    ),
}

# Inverse spherical-tensor to Cartesian transformation matrix

UMAT_SPHER_TO_CART = {
    key: np.linalg.pinv(val) for key, val in UMAT_CART_TO_SPHER.items()
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