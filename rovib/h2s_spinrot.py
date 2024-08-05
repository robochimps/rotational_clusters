"""Ab initio calculated spin-rotation constants for H2S"""

import numpy as np
from scipy import constants

BOHR_TO_ANGSTROM = constants.physical_constants["Bohr radius"][0] * 1e10

# Extract from the CFOUR output for r1 = 1.35 (Angstrom), r2 = 1.35, and alpha = 92.0
#   calculated using CCSD(T) with ACV(T+d)Z basis on S atom and AVTZ basis on H atoms

# Total Spin-Rotation Tensor (kHz)
#
#                  Jx          Jy          Jz
#
#   S #1    x    0.000000    0.000000    0.000000
#   S #1    y    0.000000    0.000000    0.000000
#   S #1    z    0.000000    0.000000    0.000000
#
#   H #21   x   16.747332    0.000000    0.000000
#   H #21   y    0.000000   17.422740  -18.340862
#   H #21   z    0.000000  -20.255678   13.383626
#
#   H #22   x   16.747332    0.000000    0.000000
#   H #22   y    0.000000   17.422740   18.340862
#   H #22   z    0.000000   20.255678   13.383626


# Reference Cartesian coordinates of atoms (in Bohr)
#
# S        16         0.00000000     0.00000000     0.10509864
# H         1        -0.00000000    -1.83512992    -1.66706573
# H         1        -0.00000000     1.83512992    -1.66706573

_COORDS = [1.35, 1.35, 92.0 * np.pi / 180]

_ATOM_XYZ = (
    np.array(
        [
            [0.00000000, 0.00000000, 0.10509864],
            [-0.00000000, -1.83512992, -1.66706573],
            [-0.00000000, 1.83512992, -1.66706573],
        ]
    )
    * BOHR_TO_ANGSTROM
)

_SR_H1_XYZ = np.array(
    [
        [16.747332, 0.000000, 0.000000],
        [0.000000, 17.422740, -18.340862],
        [0.000000, -20.255678, 13.383626],
    ]
)

_SR_H2_XYZ = np.array(
    [
        [16.747332, 0.000000, 0.000000],
        [0.000000, 17.422740, 18.340862],
        [0.000000, 20.255678, 13.383626],
    ]
)


def _to_bisector_frame(atom_xyz, tens):
    """Transform Cartesian projections of a tensor and reference
    Cartesian coordinates of atoms to the bisector frame
    """
    r1 = np.linalg.norm(atom_xyz[1, :] - atom_xyz[0, :])
    r2 = np.linalg.norm(atom_xyz[2, :] - atom_xyz[0, :])
    alpha = np.arccos(
        np.dot(
            (atom_xyz[1, :] - atom_xyz[0, :]) / r1,
            (atom_xyz[2, :] - atom_xyz[0, :]) / r2,
        )
    )
    e1 = (atom_xyz[1, :] - atom_xyz[0, :]) / r1
    e2 = (atom_xyz[2, :] - atom_xyz[0, :]) / r2
    n1 = e1 + e2
    n1 = n1 / np.linalg.norm(n1)
    n2 = np.cross(e2, n1)
    n2 = n2 / np.linalg.norm(n2)
    n3 = np.cross(n2, n1)
    n3 = n3 / np.linalg.norm(n3)
    tmat = np.stack((n1, n2, n3))

    return np.dot(np.dot(tmat, tens), tmat.T), np.dot(atom_xyz, tmat.T)


def spinrot_const_bisector(coords, return_atom_xyz: bool = False):
    """Returns spin-rotation tensors for H1 and H2 atoms (in kHz)
    together with Cartesian coordinates of atoms (in Angstrom)
    in the bisector frame, calculated for reference internal
    coordinates r1 = 1.35 (Angstrom), r2 = 1.35, and alpha = 92.0
    """
    sr1, atom_xyz = _to_bisector_frame(_ATOM_XYZ, _SR_H1_XYZ)
    sr2, atom_xyz = _to_bisector_frame(_ATOM_XYZ, _SR_H2_XYZ)
    sr = np.array([sr1, sr2])
    sr = sr[None, :].repeat(len(coords), axis=0)
    if return_atom_xyz:
        return sr, atom_xyz
    else:
        return sr