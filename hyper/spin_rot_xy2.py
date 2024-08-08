from collections import defaultdict

import numpy as np
from py3nj import wigner3j, wigner6j
from scipy import constants

from ..rovib import c2v


KHZ_TO_INVCM = 1.0 / constants.value("speed of light in vacuum") * 10


def spin_rot_xy2(
    f_angmom: float,
    rovib_states,
    rovib_sr1_me,
    rovib_sr2_me,
    spin_states={0: "B2", 1: "A1"},
    spins=(0.5, 0.5),
    allowed_sym=("B1", "B2"),
    tol: float = 1e-14,
):
    omega = np.array([0, 1, 2])
    two_omega = omega * 2
    omega_coef = np.array([1, -np.sqrt(3), np.sqrt(5)])

    j_list = set([j for (j, *_) in rovib_states])

    # precompute reduced matrix elements of spin operators
    spin_me = {}
    for spin1 in spin_states.keys():
        for spin2 in spin_states.keys():
            spin_me[(spin1, spin2)] = spin_reduced_me_xy2(
                spin1, spin2, spins[0], spins[1]
            )  ### spin_me[(spin1,spin2)].shape = (2,)

    # precompute J-dependent part of the expression
    j_me = defaultdict(float)
    for j1 in j_list:
        for j2 in j_list:
            prefac = np.sqrt((2 * j1 + 1) * (2 * j2 + 1)) * omega_coef
            two_j1 = int(j1 * 2)
            two_j2 = int(j2 * 2)

            threej1 = wigner3j(
                two_j2, 2, two_j2, -two_j2, 0, two_j2, ignore_invalid=True
            )
            threej2 = wigner3j(
                two_j1, 2, two_j1, -two_j1, 0, two_j1, ignore_invalid=True
            )

            sixj1 = wigner6j(
                two_omega, [2] * 3, [2] * 3, [two_j2] * 3, [two_j1] * 3, [two_j2] * 3
            )
            sixj2 = wigner6j(
                [2] * 3, two_omega, [2] * 3, [two_j2] * 3, [two_j1] * 3, [two_j1] * 3
            )

            if abs(threej1) > tol:
                j_me[(j1, j2)] += (-1) ** omega * j2 * sixj1 / threej1 * prefac
            if abs(threej2) > tol:
                j_me[(j1, j2)] += j1 * sixj2 / threej2 * prefac
            ### j_me[(j1, j2)].shape = (3,)

    # prepare spin-rotation basis
    hyper_states = {(f_angmom, sym): [] for sym in allowed_sym}
    for j, rovib_sym, id, enr, qua in rovib_states:
        for spin, spin_sym in spin_states.items():
            tot_sym = c2v.C2V_PRODUCT_TABLE[(rovib_sym, spin_sym)]
            if tot_sym in allowed_sym:
                for f in range(abs(j - spin), j + spin + 1):
                    if f == f_angmom:
                        hyper_states[tot_sym].append(
                            [j, rovib_sym, id, enr, qua, spin, spin_sym]
                        )

    # build the total Hamiltonian matrix and compute eigenvalues
    enr = {}
    vec = {}
    for sym in allowed_sym:
        no_states = len(hyper_states[sym])
        hmat = np.zeros((no_states, no_states), dtype=np.float64)
        for i1, hyper1 in enumerate(hyper_states[sym]):
            j1, rovib_sym1, id1, enr1, qua1, spin1, spin_sym1 = hyper1
            for i2, hyper2 in enumerate(hyper_states[sym]):
                j2, rovib_sym2, id2, enr2, qua2, spin2, spin_sym2 = hyper2
                sr1 = rovib_sr1_me[(j1, j2)][(rovib_sym1, rovib_sym2)][:, id1, id2]
                sr2 = rovib_sr2_me[(j1, j2)][(rovib_sym1, rovib_sym2)][:, id1, id2]
                me = (
                    0.5
                    * (-1) ** (spin2 + f_angmom)
                    * wigner6j(spin1 * 2, j1 * 2, f_angmom * 2, j2 * 2, spin2 * 2, 2)
                    * (
                        np.dot(sr1, j_me[(j1, j2)]) * spin_me[(spin1, spin2)][0]
                        + np.dot(sr2, j_me[(j1, j2)]) * spin_me[(spin1, spin2)][1]
                    )
                )
                if i1 == i2:
                    hmat[i1, i2] += enr1
                else:
                    hmat[i1, i2] += me * KHZ_TO_INVCM
        enr[sym], vec[sym] = np.linalg.eigh(hmat)


def spin_reduced_me_xy2(I1: float, I2: float, Ia1: float, Ia2: float, tol=1e-14):
    """Computes reduced matrix element <I' || I_n^{(1)} || I> (n=1,2)
    of nuclear spin operator for two-spin system.

    Args:
        I1, I2 (float): bra and ket spin quantum numbers, (i.e., I' and I in the above formula)
        Ia1, Ia2 (float): nuclear spins of atoms corresponding to n=1 and 2

    Returns:
        Array with two elements for n=1 and 2
    """
    two_I1 = int(I1 * 2)
    two_I2 = int(I2 * 2)
    two_Ia1 = int(Ia1 * 2)
    two_Ia2 = int(Ia2 * 2)
    prefac = np.sqrt((2 * I1 + 1) * (2 * I2 + 1))
    prefac1 = (-1) ** (Ia1 + Ia2 + I2 + 1) * Ia1 * prefac
    prefac2 = (-1) ** (Ia1 + Ia2 + I1 + 1) * Ia2 * prefac
    threej1 = wigner3j(two_Ia1, 2, two_Ia1, -two_Ia1, 0, two_Ia1, ignore_invalid=True)
    threej2 = wigner3j(two_Ia2, 2, two_Ia2, -two_Ia2, 0, two_Ia2, ignore_invalid=True)
    sixj1 = wigner6j(two_Ia1, two_I1, two_Ia2, two_I2, two_Ia1, 2, ignore_invalid=True)
    sixj2 = wigner6j(two_Ia2, two_I1, two_Ia1, two_I2, two_Ia2, 2, ignore_invalid=True)
    me = np.zeros(2, dtype=np.float64)
    if np.abs(threej1) > tol:
        me[0] = prefac1 * sixj1 / threej1
    if np.abs(threej2) > tol:
        me[1] = prefac2 * sixj2 / threej2
    return me


if __name__ == "__main__":
    Ia1 = 1 / 2
    Ia2 = 1 / 2

    for I1 in (0, 1):
        for I2 in (0, 1):
            me = spin_reduced_me_xy2(I1, I2, Ia1, Ia2)
            print(I1, I2, me)