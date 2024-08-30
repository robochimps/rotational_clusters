"""Spin-rotation interaction Hamiltonian and hyperfine solutions
for an XY2-type triatomic molecule
"""

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from py3nj import wigner3j, wigner6j
from scipy import constants

from .c2v import C2V_PRODUCT_TABLE
from .cartens import SPHER_IND, UMAT_SPHER_TO_CART

KHZ_TO_INVCM = 1.0 / constants.value("speed of light in vacuum") * 10


def dipole_xy2(
    qua: Dict[float, Dict[str, np.ndarray]],
    vec: Dict[float, Dict[str, np.ndarray]],
    rovib_dipole_me: Dict[Tuple[int, int], Dict[Tuple[str, str], np.ndarray]],
):

    rank = 1

    # compute M-tensor

    mmat = {}
    for f1 in qua.keys():
        for f2 in qua.keys():

            m1 = np.linspace(-f1, -f1, int(2 * f1) + 1)
            m2 = np.linspace(-f2, -f2, int(2 * f2) + 1)
            m12 = np.concatenate(
                (
                    m1[:, None, None].repeat(len(m2), axis=1),
                    m2[None, :, None].repeat(len(m1), axis=0),
                ),
                axis=-1,
            ).reshape(-1, 2)
            n = len(m12)
            m12_1, m12_2 = m12.T

            p = f1 - m12_1
            ip = p.astype(int)
            assert np.all(
                abs(p - ip) < 1e-16
            ), f"f1 - m1: {f1} - {m12_1} is not an integer number"
            prefac = (-1) ** ip * np.sqrt((2 * f1 + 1) * (2 * f2 + 1))

            threej = [
                wigner3j(
                    [f1 * 2] * n,
                    [omega] * n,
                    [f2 * 2] * n,
                    -m12_1 * 2,
                    [sigma * 2] * n,
                    m12_2 * 2,
                ).reshape(len(m1), len(m2))
                for (omega, sigma) in SPHER_IND[rank]
            ]
            mmat[(f1, f2)] = prefac * np.einsum(
                "skl,cs->klc",
                threej,
                UMAT_SPHER_TO_CART[rank],
                optimize="optimal",
            )

    # compute K-tensor

    hmat = []
    for f1 in qua.keys():
        for sym1 in qua[f1].keys():
            vec1 = vec[f1][sym1]

            hrow = []
            for f2 in qua.keys():
                for sym2 in qua[f2].keys():
                    vec2 = vec[f2][sym2]

                    kmat = []
                    for j1, rov_sym1, i1, spin_sym1, rov_qua1 in qua[f1][sym1]:
                        krow = []
                        for j2, rov_sym2, i2, spin_sym2, rov_qua2 in qua[f2][sym2]:

                            if i1 == i2:
                                p = i2 + f2 + j1 + j2
                                ip = int(p)
                                assert (
                                    abs(p - ip) < 1e-16
                                ), f"I2 + F2 + J1 + J2: {i2} + {f2} + {j1} + {j2} is not an integer number"
                                prefac = (
                                    (-1) ** ip
                                    * np.sqrt((2 * j1 + 1) * (2 * j2 + 1))
                                    * wigner6j(
                                        j1 * 2,
                                        int(f1 * 2),
                                        int(i2 * 2),
                                        int(f2 * 2),
                                        int(j2 * 2),
                                        2,
                                    )
                                )
                                try:
                                    me = rovib_dipole_me[(j1, j2)][(rov_sym1, rov_sym2)]
                                except KeyError:
                                    me = 0
                            else:
                                me = 0

                            if isinstance(me, np.ndarray):
                                krow.append(me)
                            else:
                                krow.append(np.zeros((len(rov_qua1), len(rov_qua2))))

                        kmat.append(krow)

                    # transform dipole matrix elements to hyperfine basis
                    kmat = np.einsum(
                        "ik,ij,jl->kl", np.conj(vec1), kmat, vec2, optimize="optimal"
                    )

                    hrow.append(kmat * mmat[(f1, f2)])
            hmat.append(hrow)


def spinrot_xy2(
    f_angmom: float,
    rovib_enr_invcm: Dict[int, Dict[str, np.ndarray]],
    rovib_qua: Dict[int, Dict[str, np.ndarray]],
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
        rovib_qua (dict): Rovibrational state assignments for different values
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
            contains (J, rov_sym, I, spin_sym, *rov_qua).
            Here, J is the rotational quantum number, rov_sym is the symmetry of the rovibrational
            state, I is the spin quantum number, spin_sym denotes the symmetry of the spin state,
            and rov_qua is the set of rovibrational quanta provided at the input,
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
    qua = {}

    for sym in allowed_sym:
        hmat = []

        for i1, qua1 in enumerate(quanta[sym]):
            j1, rov_sym1, spin1, spin_sym1 = qua1
            rov_enr1 = rovib_enr_invcm[j1][rov_sym1]
            hrow = []

            for i2, qua2 in enumerate(quanta[sym]):
                j2, rov_sym2, spin2, spin_sym2 = qua2
                rov_enr2 = rovib_enr_invcm[j2][rov_sym2]

                p = spin2 + f_angmom
                ip = int(p)
                assert (
                    abs(p - ip) < 1e-16
                ), f"spin2 + f_angmom: {spin2} + {f_angmom} = {p} is not an integer number"

                prefac = (
                    0.5
                    * (-1) ** ip
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

        qua[sym] = np.concatenate(
            [
                np.concatenate(
                    (
                        np.repeat(j, len(rovib_qua[j][rov_sym]))[:, None],
                        np.repeat(rov_sym, len(rovib_qua[j][rov_sym]))[:, None],
                        np.repeat(i, len(rovib_qua[j][rov_sym]))[:, None],
                        np.repeat(spin_sym, len(rovib_qua[j][rov_sym]))[:, None],
                        rovib_qua[j][rov_sym],
                    ),
                    axis=-1,
                )
                for (j, rov_sym, i, spin_sym) in quanta[sym]
            ],
            axis=0,
        )

    return enr, vec, qua


def spin_reduced_me_xy2(
    I1: float, I2: float, Ia1: float, Ia2: float, tol=1e-14
) -> NDArray[np.float64]:
    """Computes the reduced matrix elements <I' || I_n^{(1)} || I>
    of the nuclear spin operators I_n (n=1, 2) for a two-spin system.

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

    p1 = Ia1 + Ia2 + I2 + 1
    ip1 = int(p1)
    assert (
        abs(p1 - ip1) < 1e-16
    ), f"Ia1 + Ia2 + I2 + 1: {Ia1} + {Ia2} + {I2} + 1 = {p1} is not an integer number"

    p2 = Ia1 + Ia2 + I1 + 1
    ip2 = int(p2)
    assert (
        abs(p2 - ip2) < 1e-16
    ), f"Ia1 + Ia2 + I1 + 1: {Ia1} + {Ia2} + {I1} + 1 = {p2} is not an integer number"

    prefac1 = (-1) ** ip1 * Ia1 * prefac
    prefac2 = (-1) ** ip2 * Ia2 * prefac
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


def spin_me_xy2(
    I1: float, I2: float, Ia1: float, Ia2: float, tol=1e-14
) -> NDArray[np.complex128]:
    """Computes the matrix element <I', m' | I_n | I, m>
    of the nuclear spin operators I_n (n=1, 2) for a two-spin system.

    Args:
        I1, I2 (float): the bra and ket spin quantum numbers, respectively,
            i.e., I' and I in the above formula
        Ia1, Ia2 (float): the nuclear spins of atoms corresponding to n=1 and 2

    Returns:
        me (array(2*I1+1, 2*I2+1, 3, 2)): Array containing matrix elemens
            for different values of m' (ranging from -I1 to I1)
            and m (ranging from -I2 to I2). The third dimension (3) is the x, y, or z
            component of I_n and the fourth - the 'n' index of the nucleus.
    """
    rme = spin_reduced_me_xy2(I1, I2, Ia1, Ia2, tol=tol)
    me = np.zeros((int(2 * I1) + 1, int(2 * I2) + 1, 3, 2), dtype=np.complex128)

    two_I1 = int(I1 * 2)
    two_I2 = int(I2 * 2)

    for im1, m1 in enumerate(np.linspace(-I1, I1, int(2 * I1) + 1)):
        two_m1 = int(m1 * 2)

        for im2, m2 in enumerate(np.linspace(-I2, I2, int(2 * I2) + 1)):
            two_m2 = int(m2 * 2)

            p = I1 - m1
            ip = int(p)
            assert (
                abs(p - ip) < 1e-16
            ), f"I1 - m1: {I1} - {m1} = {p} is not an integer number"

            threej = wigner3j(
                [two_I1] * 3,
                [2, 2, 2],
                [two_I2] * 3,
                [-two_m1] * 3,
                [-2, 0, 2],
                [two_m2] * 3,
                ignore_invalid=True,
            )
            prefac = (-1) ** ip * np.dot(UMAT_SPHER_TO_CART[1], threej)
            me[im1, im2] = prefac[:, None] * rme[None, :]
    return me