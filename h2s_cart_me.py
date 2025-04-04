import jax

jax.config.update("jax_platform_name", "cpu")

from typing import Dict, List

import h5py
import numpy as np
from jax import config
from jax import numpy as jnp
from rovib.cartens import CART_IND
from rovib.symtop import threej_wang, rotme_ovlp

config.update("jax_enable_x64", True)


def read_cluster_state_ind(
    min_j: int = 40, max_j: int = 60, pmax: int = 24
) -> Dict[int, Dict[str, List[int]]]:
    """Reads indices of rotational cluster states from `h2s_cluster_states_id.txt` file.

    Args:
        min_j (int): Minimal value of rotational quantum number J for reading cluster states.
        max_j (int): Maximal value of rotational quantum number J for reading cluster states.

    Returns:
        (dict of {int: dict of {str: list of int}}): A nested dictionary containing
            list of cluster state indices. The structure is: `state_ind[j][sym]`, where `j`
            (int) is the rotational quantum number and `sym` (str) is a symmetry label
            in the C2v group.
    """
    state_ind = {}
    with open(f"h2s_cluster_states_id_pmax{pmax}.txt", "r", encoding="utf-8") as fl:
        for line in fl:
            w = line.split()
            j = [int(w[0 + i * 12]) for i in range(4)]
            id = [int(w[1 + i * 12]) for i in range(4)]
            sym = [w[3 + i * 12] for i in range(4)]
            assert all(j[0] == elem for elem in j), f"Not all J quanta are equal: {j}"
            j_ = j[0]

            if j_ >= min_j and j_ <= max_j:
                state_ind[j_] = {}
                for sym_, id_ in zip(sym, id):
                    try:
                        state_ind[j_][sym_].append(id_)
                    except KeyError:
                        state_ind[j_][sym_] = [id_]
                state_ind[j_] = dict(sorted(state_ind[j_].items()))
                print(f"J = {j_}, cluster-state IDs: {state_ind[j_]}")
    return state_ind


def run_rovib_enr(
    out_filename: str, state_ind_list: Dict[int, Dict[str, List[int]]], pmax: int = 24
):
    with h5py.File(out_filename, "w") as h5:
        for j in state_ind_list.keys():
            enr, qua, vind, rind, coefs = _rovib_states(
                j, state_ind_list=state_ind_list[j], pmax=pmax
            )
            print(
                f"store energies for J = {j}, states = {[(sym, np.round(val, 6)) for sym, val in enr.items()]}"
            )
            for sym, e in enr.items():
                qua_str = [",".join(elem) for elem in qua[sym]]
                max_len = max([len(elem) for elem in qua_str])
                qua_ascii = [elem.encode("ascii", "ignore") for elem in qua_str]
                h5.create_dataset(f"energies/{j}/{sym}", data=e)
                h5.create_dataset(f"coefficients/{j}/{sym}", data=coefs[sym])
                h5.create_dataset(f"vib-indices/{j}/{sym}", data=vind[sym])
                h5.create_dataset(f"rot-indices/{j}/{sym}", data=rind[sym])
                h5.create_dataset(
                    f"quanta/{j}/{sym}",
                    (len(qua_ascii), 1),
                    f"S{max_len}",
                    data=qua_ascii,
                )


def run_rovib_me(
    j1: int,
    j2: int,
    out_filename: str,
    state_ind_list: Dict[int, Dict[str, List[int]]] = None,
    pmax: int = 24,
    verbose: bool = True,
):
    """
    Computes and stores dipole and spin-rotation rovibrational matrix elements.

    The matrix elements are calculated between states with rotational quantum numbers
    `j1` (bra) and `j2` (ket), and saved into the specified HDF5 output file.

    Args:
        j1 (int): Rotational quantum number J for the bra state.
        j2 (int): Rotational quantum number J for the ket state.
        out_filename (str): Name of the output file where the matrix elements will be stored.
        state_ind_list (dict of {int: dict of {str: list of int}}): A nested dictionary
            specifying a subset of rovibrational states for which the matrix elements
            are calculated. The structure should be: `state_ind_list[j][sym]`, where `j`
            (int) is the rotational quantum number and `sym` (str) is a symmetry label
            in the C2v group. If `None` (default), matrix elements are calculated for all states.
        pmax (int): Vibrational polyad number. Used for defining file names containing precomputed
            rovibrational wavefunctions and vibrational matrix elements. Default is 24.
        verbose (bool): If `True`, matrix elements will be printed into output. Default is `True`.

    Returns:
        None
    """
    with h5py.File(f"h2s_vibme_pmax{pmax}.h5", "r") as h5:
        sr_h1_vib = h5["spin-rotation"]["h1"][:]
        sr_h2_vib = h5["spin-rotation"]["h2"][:]
        dipole_vib = h5["dipole-moment"][:]
        coords_vib = h5["coordinate"][:]

    dj = abs(j1 - j2)

    if dj == 0:
        coords_me = _rovib_me(
            j1,
            coords_vib,
            state_ind_list=state_ind_list[j1],
            pmax=pmax,
        )
    else:
        coords_me = {}

    if dj <= 1:
        dipole_me = _tensor_rovib_me(
            1,
            j1,
            j2,
            dipole_vib,
            state_ind_list1=state_ind_list[j1],
            state_ind_list2=state_ind_list[j2],
            pmax=pmax,
        )
    else:
        dipole_me = {}

    if dj <= 2:
        sr1_me = _tensor_rovib_me(
            2,
            j1,
            j2,
            sr_h1_vib,
            state_ind_list1=state_ind_list[j1],
            state_ind_list2=state_ind_list[j2],
            pmax=pmax,
        )
        sr2_me = _tensor_rovib_me(
            2,
            j1,
            j2,
            sr_h2_vib,
            state_ind_list1=state_ind_list[j1],
            state_ind_list2=state_ind_list[j2],
            pmax=pmax,
        )
    else:
        sr1_me = {}
        sr2_me = {}

    if coords_me or dipole_me or sr1_me or sr2_me:
        print(f"store matrix elements for J1 = {j1} J2 = {j2} in file {out_filename}")
        with h5py.File(out_filename, "w") as h5:
            for label, oper in zip(
                ("coordinate", "dipole", "spin-rotation-H1", "spin-rotation-H2"),
                (coords_me, dipole_me, sr1_me, sr2_me),
            ):
                for (sym1, sym2), me in oper.items():
                    if verbose:
                        print(
                            f"{label}, sym = {(sym1, sym2)}, shape = {me.shape}, val = {me}"
                        )
                    h5.create_dataset(f"{label}/{sym1}/{sym2}", data=me)


def _rovib_states(j: int, state_ind_list: Dict[str, List[int]] = None, pmax: int = 24):
    """Reads rovibrational energies and quanta"""
    h5 = h5py.File(f"pmax{pmax}/h2s_coefficients_pmax{pmax}_j{j}.h5", "r")
    energies = {}
    quanta = {}
    vib_indices = {}
    rot_indices = {}
    coefficients = {}
    for sym in h5["energies"].keys():
        enr = h5["energies"][sym][:]
        coefs = h5["coefficients"][sym][:]
        vind = h5["vib-indices"][sym][:]
        rind = h5["rot-indices"][sym][:]
        qua = np.array(
            [elem[0].decode("utf-8").split(",") for elem in h5["quanta"][sym][:]]
        )
        if state_ind_list is not None and sym in state_ind_list:
            energies[sym] = enr[state_ind_list[sym]]
            quanta[sym] = qua[state_ind_list[sym]]
            coefficients[sym] = coefs[:, state_ind_list[sym]]
        else:
            energies[sym] = enr
            quanta[sym] = qua
            coefficients[sym] = coefs
        vib_indices[sym] = vind
        rot_indices[sym] = rind
    h5.close()
    return energies, quanta, vib_indices, rot_indices, coefficients


def _tensor_rovib_me(
    rank: int,
    j1: int,
    j2: int,
    vib_me: np.ndarray,
    state_ind_list1: Dict[str, List[int]] = None,
    state_ind_list2: Dict[str, List[int]] = None,
    pmax: int = 24,
    linear: bool = False,
):
    """Computes rovibrational matrix elements of a laboratory-frame Cartesian tensor operator

    The first two dimensions of `vib_me` array correspond to the indices of bra and ket vibrational states,
    `vib_me[:, :, i]` with i = 0..2 for x, y, z components of rank-1 tensor,
    `vib_me[:, :, i, j]` with i, j = 0..2 for xx, xy, xz, yz, ..., zz components of rank-2 tensor.

    `state_ind_list1` and `state_ind_list2` are dictionaries containing lists of rovibrational state
    indices for the target bra and ket states, as dictionary values, for different state symmetries,
    as dictionary keys.
    If `state_ind_list1` or `state_ind_list2` is None, the matrix elements for all bra or ket
    rovibrational states for given bra J=`j1` and ket J=`j2` will be computed.
    """

    # determine the order of Cartesian indices in the Cartesian-to-spherical tensor
    #   transformation matrix (in cartens.CART_IND and symtop.threej_wang)
    cart_ind = [["xyz".index(x) for x in elem] for elem in CART_IND[rank]]

    # reshape vibrational matrix elements such that the order of Cartesian indices
    #   correspond to the order in symtop.threej_wang output
    if rank == 1:
        vib_me2 = jnp.moveaxis(jnp.array([vib_me[:, :, i] for (i,) in cart_ind]), 0, -1)
    elif rank == 2:
        vib_me2 = jnp.moveaxis(
            jnp.array([vib_me[:, :, i, j] for (i, j) in cart_ind]), 0, -1
        )
    else:
        raise ValueError(
            f"Index mapping for tensor of rank = {rank} is not implemented"
        )

    # compute rotational matrix elements of three-j symbol contracted with
    #   Cartesian-to-spherical tensor transformation matrix
    jktau_list1, jktau_list2, rot_me = threej_wang(rank, j1, j2, linear=linear)
    # rot_me[omega].shape = (2*j1+1, 2*j2+1, ncart)

    h5_1 = h5py.File(f"pmax{pmax}/h2s_coefficients_pmax{pmax}_j{j1}.h5", "r")
    h5_2 = h5py.File(f"pmax{pmax}/h2s_coefficients_pmax{pmax}_j{j2}.h5", "r")

    res = {}

    for sym1 in h5_1["energies"].keys():
        enr1 = h5_1["energies"][sym1][:]
        coefs1 = h5_1["coefficients"][sym1][:]
        vind1 = h5_1["vib-indices"][sym1][:]
        rind1 = h5_1["rot-indices"][sym1][:]
        qua1 = np.array(
            [elem[0].decode("utf-8").split(",") for elem in h5_1["quanta"][sym1][:]]
        )

        if state_ind_list1 is not None:
            if sym1 in state_ind_list1:
                ind = state_ind_list1[sym1]
                enr1 = enr1[ind]
                coefs1 = coefs1[:, ind]
            else:
                continue

        for sym2 in h5_2["energies"].keys():
            enr2 = h5_2["energies"][sym2][:]
            coefs2 = h5_2["coefficients"][sym2][:]
            vind2 = h5_2["vib-indices"][sym2][:]
            rind2 = h5_2["rot-indices"][sym2][:]
            qua2 = np.array(
                [elem[0].decode("utf-8").split(",") for elem in h5_2["quanta"][sym2][:]]
            )

            if state_ind_list2 is not None:
                if sym2 in state_ind_list2:
                    ind = state_ind_list2[sym2]
                    enr2 = enr2[ind]
                    coefs2 = coefs2[:, ind]
                else:
                    continue

            vib_me_ = vib_me2[jnp.ix_(vind1, vind2)]
            me = []
            for omega in rot_me.keys():
                me_ = jnp.einsum(
                    "ijc,ijc->ij", vib_me_, rot_me[omega][jnp.ix_(rind1, rind2)]
                )
                me.append(jnp.einsum("ik,ij,jl->kl", jnp.conj(coefs1), me_, coefs2))
            res[(sym1, sym2)] = jnp.moveaxis(jnp.array(me), 0, -1)
    h5_1.close()
    h5_2.close()
    return res


def _rovib_me(
    j: int,
    vib_me: np.ndarray,
    state_ind_list: Dict[str, List[int]] = None,
    pmax: int = 20,
    linear: bool = False,
):
    """Computes rovibrational matrix elements of a molecular-frame operator

    The first two dimensions of vib_me array correspond to indices of bra and ket vibrational states.

    `state_ind_list` is a dictionary containing lists of rovibrational state indices for target states,
    as dictionary values, for different state symmetries as dictionary keys.
    If `state_ind_list` is None, matrix elements for all states for given J=`j` will be computed.
    """

    rot_me, *_ = rotme_ovlp(j, linear=linear)

    h5 = h5py.File(f"pmax{pmax}/h2s_coefficients_pmax{pmax}_j{j}.h5", "r")

    res = {}

    for sym1 in h5["energies"].keys():
        enr1 = h5["energies"][sym1][:]
        coefs1 = h5["coefficients"][sym1][:]
        vind1 = h5["vib-indices"][sym1][:]
        rind1 = h5["rot-indices"][sym1][:]
        qua1 = np.array(
            [elem[0].decode("utf-8").split(",") for elem in h5["quanta"][sym1][:]]
        )

        if state_ind_list is not None:
            if sym1 in state_ind_list:
                ind = state_ind_list[sym1]
                enr1 = enr1[ind]
                coefs1 = coefs1[:, ind]
            else:
                continue

        for sym2 in h5["energies"].keys():
            enr2 = h5["energies"][sym2][:]
            coefs2 = h5["coefficients"][sym2][:]
            vind2 = h5["vib-indices"][sym2][:]
            rind2 = h5["rot-indices"][sym2][:]
            qua2 = np.array(
                [elem[0].decode("utf-8").split(",") for elem in h5["quanta"][sym2][:]]
            )

            if state_ind_list is not None:
                if sym2 in state_ind_list:
                    ind = state_ind_list[sym2]
                    enr2 = enr2[ind]
                    coefs2 = coefs2[:, ind]
                else:
                    continue

            res[(sym1, sym2)] = jnp.einsum(
                "ik,ij...,ij,jl->kl...",
                jnp.conj(coefs1),
                vib_me[jnp.ix_(vind1, vind2)],
                rot_me[jnp.ix_(rind1, rind2)],
                coefs2,
            )

    h5.close()
    return res


if __name__ == "__main__":
    import sys

    PMAX = 24

    # indices of cluster states
    state_ind = read_cluster_state_ind(pmax=PMAX)
    out_filename = "cluster"

    # ... alternatively indices of the lowest 10 states
    # NO_STATES = 1000
    # out_filename = f"{NO_STATES}"
    # state_ind = {
    #     j: {
    #         "A1": list(range(NO_STATES)),
    #         "A2": list(range(NO_STATES)),
    #         "B1": list(range(NO_STATES)),
    #         "B2": list(range(NO_STATES)),
    #     }
    #     for j in range(40, 61)
    # }

    try:
        # compute and store matrix elements
        j1 = int(sys.argv[1])
        j2 = int(sys.argv[2])
        out_filename = f"h2s_me_{out_filename}_j{j1}_j{j2}_pmax{PMAX}.h5"
        run_rovib_me(
            j1, j2, out_filename, state_ind_list=state_ind, verbose=True, pmax=PMAX
        )

    except IndexError:
        # store energies
        min_j = min(list(state_ind.keys()))
        max_j = max(list(state_ind.keys()))
        out_filename = f"h2s_enr_{out_filename}_j{min_j}_j{max_j}_pmax{PMAX}.h5"
        run_rovib_enr(out_filename, state_ind, pmax=PMAX)