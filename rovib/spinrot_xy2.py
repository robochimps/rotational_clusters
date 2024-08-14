"""Spin-rotation interaction Hamiltonian and hyperfine solutions
for an XY2-type triatomic molecule
"""

from typing import Dict, List, Tuple

import numpy as np
from py3nj import wigner3j, wigner6j
from scipy import constants

from .c2v import C2V_PRODUCT_TABLE

KHZ_TO_INVCM = 1.0 / constants.value("speed of light in vacuum") * 10


def spinrot_xy2(
    f_angmom: float,
    rovib_enr_invcm: Dict[int, Dict[str, np.ndarray]],
    rovib_sr1_me_khz: Dict[Tuple[int, int], Dict[Tuple[str, str], np.ndarray]],
    rovib_sr2_me_khz: Dict[Tuple[int, int], Dict[Tuple[str, str], np.ndarray]],
    spin_states: List[Tuple[int, str]] = [(0, "B2"), (1, "A1")],
    spins: List[float] = [0.5, 0.5],
    allowed_sym: List[str] = ["B1", "B2"],
    tol: float = 1e-14,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, List[Tuple[float, str, float, str]]],
]:
    """
    Computes spin-rotation hyperfine energies and wavefunctions for an XY2-type triatomic molecule.

    See `h2s_hyperfine.py` for the use example.

    Args:
        f_angmom (float): Quantum number corresponding to the total angular momentum F = J + I,
            where J is the rotational angular momentum and I is the nuclear spin angular momentum.
        rovib_enr_invcm (dict): Rovibrational energies (in cm^-1) for different values
            of rotational quantum number J and C2v symmetry.
        rovib_sr1_me_khz (dict): Matrix elements (in kHz) of the spin-rotation tensor
            for atom Y1 in the XY2 molecule.
        rovib_sr2_me_khz (dict): Matrix elements (in kHz) of the spin-rotation tensor
            for atom Y2 in the XY2 molecule.
        spin_states (list): A list containing spin states, where each element represents
            the total spin quantum number I and its corresponding symmetry in the C2v point group.
        spins (list of float): A list containing the spins of atoms Y1 and Y1, respectively.
        allowed_sym (list): A list of symmetry labels that are allowed for spin-rovibrational
            states according to the Pauli exclusion principle.
        tol (float): Tolerance level for treating certain three-j symbols as zero.

    Returns:
        enr (dict): Hyperfine energies for each symmetry label specified in `allowed_sym`.
        vec (dict): Eigenvectors corresponding to the hyperfine energies for each symmetry.
        qua (dict): Quantum number assignments for each symmetry, where each tuple
            contains (J, rov_sym, I, spin_sym). Here, J and I are the rotational and spin quantum
            numbers, respectively, and rov_sym and spin_sym denote the symmetries of the rovibrational
            and spin states.
    """
    omega = np.array([0, 1, 2])
    two_omega = omega * 2
    omega_coef = np.array([1, -np.sqrt(3), np.sqrt(5)])

    j_list = list(rovib_enr_invcm.keys())

    # precompute reduced matrix elements of spin operators

    spin_me = {}
    for spin1, spin_sym1 in spin_states:
        for spin2, spin_sym2 in spin_states:
            spin_me[(spin1, spin2)] = spin_reduced_me_xy2(
                spin1, spin2, spins[0], spins[1]
            )

    # precompute J-dependent part of the expression

    j_me = {(j1, j2): np.zeros(3, dtype=np.float64) for j1 in j_list for j2 in j_list}

    for j1 in j_list:
        for j2 in j_list:
            prefac = np.sqrt((2 * j1 + 1) * (2 * j2 + 1)) * omega_coef
            two_j1 = int(j1 * 2)
            two_j2 = int(j2 * 2)

            if j2 > 0:
                threej = wigner3j(
                    two_j2, 2, two_j2, -two_j2, 0, two_j2, ignore_invalid=True
                )
                if abs(threej) < tol:
                    raise ValueError(
                        f"Can't divide by 3j-symbol (J2 1 J2)(-J2 0 J2) = {threej}"
                    ) from None
                sixj = wigner6j(
                    two_omega,
                    [2] * 3,
                    [2] * 3,
                    [two_j2] * 3,
                    [two_j1] * 3,
                    [two_j2] * 3,
                    ignore_invalid=True,
                )
                j_me[(j1, j2)] += (-1) ** omega * j2 * sixj / threej * prefac

            if j1 > 0:
                threej = wigner3j(
                    two_j1, 2, two_j1, -two_j1, 0, two_j1, ignore_invalid=True
                )
                sixj = wigner6j(
                    [2] * 3,
                    two_omega,
                    [2] * 3,
                    [two_j2] * 3,
                    [two_j1] * 3,
                    [two_j1] * 3,
                    ignore_invalid=True,
                )
                if abs(threej) < tol:
                    raise ValueError(
                        f"Can't divide by 3j-symbol (J1 1 J1)(-J1 0 J1) = {threej}"
                    ) from None
                j_me[(j1, j2)] += j1 * sixj / threej * prefac

    # prepare spin-rovibrational basis quanta

    quanta = {sym: [] for sym in allowed_sym}
    for j in rovib_enr_invcm.keys():
        for rov_sym in rovib_enr_invcm[j].keys():
            for spin, spin_sym in spin_states:
                tot_sym = C2V_PRODUCT_TABLE[(rov_sym, spin_sym)]
                if tot_sym in allowed_sym:
                    for f in range(abs(j - spin), j + spin + 1):
                        if f == f_angmom:
                            quanta[tot_sym].append((j, rov_sym, spin, spin_sym))

    # build total Hamiltonian matrix and compute eigenvalues
    #   for different total symmetries

    enr = {}
    vec = {}

    for sym in allowed_sym:
        hmat = []

        for i1, qua1 in enumerate(quanta[sym]):
            j1, rov_sym1, spin1, spin_sym1 = qua1
            rov_enr1 = rovib_enr_invcm[j1][rov_sym1]
            hrow = []

            for i2, qua2 in enumerate(quanta[sym]):
                j2, rov_sym2, spin2, spin_sym2 = qua2
                rov_enr2 = rovib_enr_invcm[j2][rov_sym2]

                prefac = (
                    0.5
                    * (-1) ** (spin2 + f_angmom)
                    * wigner6j(
                        spin1 * 2,
                        j1 * 2,
                        f_angmom * 2,
                        j2 * 2,
                        spin2 * 2,
                        2,
                        ignore_invalid=True,
                    )
                )

                try:
                    sr1 = rovib_sr1_me_khz[(j1, j2)][(rov_sym1, rov_sym2)]
                    me1 = np.dot(sr1, j_me[(j1, j2)]) * spin_me[(spin1, spin2)][0]
                except KeyError:
                    me1 = 0

                try:
                    sr2 = rovib_sr2_me_khz[(j1, j2)][(rov_sym1, rov_sym2)]
                    me2 = np.dot(sr2, j_me[(j1, j2)]) * spin_me[(spin1, spin2)][1]
                except KeyError:
                    me2 = 0

                me = prefac * (me1 + me2) * KHZ_TO_INVCM
                if i1 == i2:
                    me += np.diag(rov_enr1)

                if isinstance(me, np.ndarray):
                    hrow.append(me)
                else:
                    hrow.append(np.zeros((len(rov_enr1), len(rov_enr2))))

            hmat.append(hrow)

        hmat = np.block(hmat)

        enr[sym], vec[sym] = np.linalg.eigh(hmat)

    return enr, vec, quanta


def spin_reduced_me_xy2(I1: float, I2: float, Ia1: float, Ia2: float, tol=1e-14):
    """Computes reduced matrix element <I' || I_n^{(1)} || I> (n=1,2)
    of nuclear spin operator for two-spin system.

    Args:
        I1, I2 (float): the bra and ket spin quantum numbers, respectively,
            i.e., I' and I in the above formula
        Ia1, Ia2 (float): the nuclear spins of atoms corresponding to n=1 and 2

    Returns:
        me (array(2)): Array with two elements for n=1 and n=2
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