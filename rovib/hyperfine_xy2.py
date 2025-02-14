"""Spin-rotation interaction Hamiltonian and hyperfine solutions
for an XY2-type triatomic molecule
"""

from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from py3nj import wigner3j, wigner6j
from scipy import constants

from .c2v import C2V_PRODUCT_TABLE
from .cartens import SPHER_IND, UMAT_CART_TO_SPHER, UMAT_SPHER_TO_CART

KHZ_TO_INVCM = 1.0 / constants.value("speed of light in vacuum") * 10


def dipole_xy2(
    qua: Dict[float, Dict[str, np.ndarray]],
    vec: Dict[float, Dict[str, np.ndarray]],
    rovib_dipole_me: Dict[Tuple[int, int], Dict[Tuple[str, str], np.ndarray]],
    m_val: float = None,
    thresh: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the matrix elements of the laboratory-frame dipole moment operator
    between hyperfine states.

    Args:
        qua (dict): Dictionary containing hyperfine state assignments for different
            values of the spin-rotational angular momentum quantum number F and C2v symmetry.
            The structure matches the `quanta_block` output of the `spinrot_xy2` function,
            that is `qua[F][sym][:]`, where each element is a tuple in the format
            (J, rov_sym, I, spin_sym, no_rov_states), where:
            - J (int): Rotational angular momentum quantum number.
            - rov_sym (str): Rovibrational-state symmetry label.
            - I (int): Nuclear spin quantum number.
            - spin_sym (str): Spin-state symmetry label.
            - no_rov_states (int): Number of rovibrational states.

        vec (dict): Dictionary containing the hyperfine eigenvectors for different values of
            F and C2v symmetry. The structure is `vec[F][sym][:, :].

        rovib_dipole_me (dict): Dictionary containing the rovibrational matrix elements of the
            dipole moment operator. The structure is:
            `rovib_dipole_me[(J1, J2)][(sym1, sym2)][istate1, istate2, omega]`, where:
            - J1, J2 (int): Rotational quantum numbers for the bra and ket rovibrational states, respectively.
            - sym1, sym2 (str): Symmetry labels for the bra and ket rovibrational states, respectively.
            - istate1, istate2 (int): Indices of the bra and ket rovibrational states.
            - omega (int or list[int]): Spherical tensor irreducible representation index.
                For rank-1 operators, omega = [1]; for rank-2 operators, omega = [0, 1, 2].

        m_f (float): Specific value of the magnetic quantum number for compuing dipole matrix elements.
            If not specified, calculations will be done for m_f = -F .. F.

        thresh (float): Threshold for neglecting matrix elements.

    Returns:
        hmat (numpy.ndarray): 3D array containing the matrix elements of the dipole moment operator.
            The first dimension (0, 1, 2) corresponds to the Cartesian components (X, Y, Z) of the
            dipole moment in the laboratory frame. The other two dimensions correspond to the bra
            and ket hyperfine states.

        coupl_qua (numpy.ndarray): Array containing quantum numbers in the dipole-coupled basis
            with resolution corresponding to hyperfine states. Each element is a tuple
            (F, m_F, sym, hyper_state_ind), where:
            - F (str): Total spin-rotational angular momentum quantum number.
            - m_F (str): Magnetic quantum number.
            - sym (str): Hyperfine-state symmetry label.
            - hyper_state_ind (str): Hyperfine-state index.
    """

    # compute M-matrix

    rank = 1
    omega_list = list(UMAT_CART_TO_SPHER[rank].keys())
    sigma_list = {
        omega: [s for (o, s) in SPHER_IND[rank] if o == omega] for omega in omega_list
    }
    sigma_ind = {
        omega: [sigma_list[omega].index(s) for (o, s) in SPHER_IND[rank] if o == omega]
        for omega in omega_list
    }

    m_list = {}
    for f in qua.keys():
        if m_val is None:
            m_list[f] = np.linspace(-f, -f, int(2 * f) + 1)
        else:
            if abs(m_val) <= f:
                m_list[f] = np.array([m_val])

    mmat = {}
    for f1 in qua.keys():
        for f2 in qua.keys():
            try:
                m1 = m_list[f1]
                m2 = m_list[f2]
            except KeyError:
                continue

            m12 = np.concatenate(
                (
                    m1[:, None, None].repeat(len(m2), axis=1),
                    m2[None, :, None].repeat(len(m1), axis=0),
                ),
                axis=-1,
            ).reshape(-1, 2)
            n = len(m12)
            m12_1, m12_2 = m12.T * 2

            p = -(m1 - f1)
            ip = p.astype(int)
            assert np.all(
                abs(p - ip) < 1e-16
            ), f"f1 - m1: {f1} - {m1} is not an integer number"
            prefac = (-1) ** ip * np.sqrt((2 * f1 + 1) * (2 * f2 + 1))

            mmat_ = []
            for omega in omega_list:
                threej = [
                    wigner3j(
                        [int(f1 * 2)] * n,
                        [omega * 2] * n,
                        [int(f2 * 2)] * n,
                        -m12_1.astype(int),
                        [sigma * 2] * n,
                        m12_2.astype(int),
                        ignore_invalid=True,
                    ).reshape(len(m1), len(m2))
                    for sigma in sigma_list[omega]
                ]
                mmat_.append(
                    np.einsum(
                        "k,skl,cs->ckl",
                        prefac,
                        threej,
                        UMAT_SPHER_TO_CART[rank][:, sigma_ind[omega]],
                        optimize="optimal",
                    )
                )
            mmat_ = np.array(mmat_)  # shape = (omega, cart, m1, m2)
            if np.any(np.abs(mmat_) > thresh):
                mmat[(f1, f2)] = np.moveaxis(mmat_, 0, 1)
                # shape = (cart, omega, m1, m2), omega = [1], cart = [0, 1, 2]

    # number of tensor Cartesian componets
    ncart = UMAT_SPHER_TO_CART[rank].shape[0]

    # compute K-matrix

    hmat = []

    for f1 in qua.keys():
        for sym1 in qua[f1].keys():
            if f1 in m_list:
                dim1 = sum(elem[4] for elem in qua[f1][sym1]) * len(m_list[f1])
            else:
                continue
            vec1 = vec[f1][sym1]

            hrow = []

            for f2 in qua.keys():
                for sym2 in qua[f2].keys():
                    if f2 in m_list:
                        dim2 = sum(elem[4] for elem in qua[f2][sym2]) * len(m_list[f2])
                    else:
                        continue
                    vec2 = vec[f2][sym2]

                    if (f1, f2) not in mmat:
                        hrow.append(np.zeros((ncart, dim1, dim2), dtype=np.complex128))
                        continue

                    kmat = []
                    for j1, rov_sym1, i1, spin_sym1, nstates1 in qua[f1][sym1]:

                        krow = []
                        for j2, rov_sym2, i2, spin_sym2, nstates2 in qua[f2][sym2]:

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
                                        ignore_invalid=True,
                                    )
                                )
                                try:
                                    me = (
                                        rovib_dipole_me[(j1, j2)][(rov_sym1, rov_sym2)]
                                        * prefac
                                    )
                                    # shape = (nstates1, nstates2, omega), omega=[1]
                                except KeyError:
                                    me = 0
                            else:
                                me = 0

                            if isinstance(me, np.ndarray):
                                krow.append(me)
                            else:
                                krow.append(
                                    np.zeros(
                                        (nstates1, nstates2, len(omega_list)),
                                        dtype=np.complex128,
                                    )
                                )

                        kmat.append(np.concatenate(krow, axis=1))
                    kmat = np.concatenate(kmat, axis=0)

                    # transform dipole matrix elements to hyperfine basis
                    kmat = np.einsum(
                        "ik,ijo,jl->okl", np.conj(vec1), kmat, vec2, optimize="optimal"
                    )

                    # multiply M and K matrices and sum over omega
                    hrow.append(
                        np.array(
                            [
                                np.sum(
                                    [np.kron(mo, ko) for mo, ko in zip(mx, kmat)],
                                    axis=0,
                                )
                                for mx in mmat[(f1, f2)]
                            ]
                        )
                    )

            hmat.append(np.concatenate(hrow, axis=-1))

    hmat = np.concatenate(hmat, axis=1)

    # assing quantum numbers

    coupl_qua = []
    for f in qua.keys():
        for sym in qua[f].keys():
            nstates = sum(elem[4] for elem in qua[f][sym])
            if f in m_list:
                coupl_qua += [
                    np.concatenate(
                        (
                            np.repeat(f, nstates)[:, None],
                            np.repeat(m, nstates)[:, None],
                            np.repeat(sym, nstates)[:, None],
                            np.arange(nstates)[:, None],
                        ),
                        axis=-1,
                    )
                    for m in m_list[f]
                ]
            else:
                continue
    coupl_qua = np.concatenate(coupl_qua, axis=0)

    return hmat, coupl_qua


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
    Dict[str, np.ndarray],
]:
    """
    Computes spin-rotation hyperfine energies and wavefunctions for an XY2-type triatomic molecule.

    Args:
        f_angmom (float): Quantum number of the total angular momentum operator F = J + I,
            where J is the rotational angular momentum and I is the nuclear-spin angular momentum.

        rovib_enr_invcm (dict): Rovibrational energies (in cm^-1) for different values
            of rotational quantum number J and C2v symmetry.
            The structure is `rovib_enr_invcm[J][sym][istate]`, where:
            - J (int): Rotational angular momentum quantum number.
            - sym (str): Rovibrational-state symmetry label.
            - istate (int): Rovibrational-state index.

        rovib_qua (dict): Dictionary containing rovibrational state assignments for different
            values of the rotational quantum number J and C2v symmetry. The structure is:
            `rovib_qua[J][sym][istate]`, where
            - J (int): Rotational angular momentum  quantum number.
            - sym (str): Rovibrational-state symmetry label.
            - istate (int): Rovibrational-state index.
            Each element is a tuple of rovibrational quantum numbers (str)
            that describe the rovibrational state.

        rovib_sr1_me_khz (dict): Dictionary containing the rovibrational matrix elements (in kHz)
            of the spin-rotation tensor for atom Y1 in the XY2 molecule. The structure is:
            `rovib_sr1_me_khz[(J1, J2)][(sym1, sym2)][istate1, istate2, omega]`, where:
            - J1, J2 (int): Rotational quantum numbers for the bra and ket rovibrational states, respectively.
            - sym1, sym2 (str): Symmetry labels for the bra and ket rovibrational states, respectively.
            - istate1, istate2 (int): Indices of the bra and ket rovibrational states.
            - omega (int or list[int]): Spherical tensor irreducible representation index.
                For rank-2 operators, omega = [0, 1, 2].

        rovib_sr2_me_khz (dict): Dictionary containing the rovibrational matrix elements (in kHz)
            of the spin-rotation tensor for atom Y2 in the XY2 molecule.

        spin_states (list of tuples): A list containing spin states. Each element is a tuple
            where the first element is the value of the total spin quantum number I
            and the second element is the corresponding spin-state symmetry in the C2v point group.
            For example, `spin_states = [(0, "B2"), (1, "A1")]`.

        spins (list of float): A list containing the spins of atoms Y1 and Y1, respectively.

        allowed_sym (list): A list of symmetry labels in the C2v group that are allowed
            for spin-rovibrational states according to the Pauli exclusion principle.

        tol (float): Tolerance level for treating certain three-j symbols as zero.

    Returns:
        enr (dict): Hyperfine energies for each symmetry label specified in `allowed_sym`.

        vec (dict): Eigenvectors corresponding to the hyperfine energies for each symmetry.

        qua (dict): Quantum numbers in the hyperfine basis for each hyperfine-state symmetry.
            Each element is a tuple (J, rov_sym, I, spin_sym, *rov_qua), where:
            - J (str): Rotational angular momentum quantum number.
            - rov_sym (str): Rovibrational-state symmetry.
            - I (str): Nuclear spin quantum number.
            - spin_sym (str): Spin-state symmetry label.
            - *rovib_qua[J][rov_sym] (tuple): Rovibrational quantum numbers.

        quanta_block (dict): Quantum numbers in the hyperfine basis for each hyperfine-state symmetry
            with resolution corresponding to rovibrational-state blocks with certain J and symmetry.
            Each element is a tuple (J, rov_sym, I, spin_sym, no_rov_states), where:
            - no_rov_states (int): Number of rovibrational states.
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

    quanta_block = {sym: [] for sym in allowed_sym}
    for j in rovib_enr_invcm.keys():
        for rov_sym in rovib_enr_invcm[j].keys():
            nstates = len(rovib_enr_invcm[j][rov_sym])
            for spin, spin_sym in spin_states:
                tot_sym = C2V_PRODUCT_TABLE[(rov_sym, spin_sym)]
                if tot_sym in allowed_sym:
                    for f in range(abs(j - spin), j + spin + 1):
                        if f == f_angmom:
                            quanta_block[tot_sym].append(
                                (j, rov_sym, spin, spin_sym, nstates)
                            )

    # build total Hamiltonian matrix and compute eigenvalues
    #   for different total symmetries

    enr = {}
    vec = {}
    qua = {}
    hout = {}

    for sym in allowed_sym:
        hmat = []

        for i1, qua1 in enumerate(quanta_block[sym]):
            j1, rov_sym1, spin1, spin_sym1, nstates1 = qua1
            rov_enr1 = rovib_enr_invcm[j1][rov_sym1]
            hrow = []

            for i2, qua2 in enumerate(quanta_block[sym]):
                j2, rov_sym2, spin2, spin_sym2, nstates2 = qua2
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
                        int(spin1 * 2),
                        j1 * 2,
                        int(f_angmom * 2),
                        j2 * 2,
                        int(spin2 * 2),
                        2,
                        ignore_invalid=True,
                    )
                )

                # shape of spin-rotation = (nstates1, nstates2, omega), omega = [0,1,2]
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

                # if i1 == i2:
                #     me += np.diag(rov_enr1)

                if isinstance(me, np.ndarray):
                    hrow.append(me)
                else:
                    hrow.append(np.zeros((nstates1, nstates2)))

            hmat.append(hrow)

        hmat = np.block(hmat)

        enr[sym], vec[sym] = np.linalg.eigh(hmat)
        hout[sym] = hmat

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
                for (j, rov_sym, i, spin_sym, nstates) in quanta_block[sym]
            ],
            axis=0,
        )

    # return enr, vec, qua, quanta_block
    return hout, qua, quanta_block


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