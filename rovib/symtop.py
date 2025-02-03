from collections import defaultdict
from collections.abc import Iterable
from typing import List, Tuple

import numpy as np
from jax import config
from jax import numpy as jnp
from py3nj import wigner3j

from .c2v import C2V_KTAU_IRREPS
from .cartens import SPHER_IND, UMAT_CART_TO_SPHER
from .wigner import _jy_eig, wigner_D

config.update("jax_enable_x64", True)


MAX_J = 100
WANG_COEFS = {}


def tabulate_wang(linear: bool):

    def _wang_coefs(j, k, tau):
        sigma = jnp.fmod(k, 3) * tau
        fac1 = pow(-1.0, sigma) / jnp.sqrt(2.0)
        fac2 = fac1 * pow(-1.0, (j + k))
        kval = [k, -k]
        if tau == 0:
            if k == 0:
                coefs = [1.0]
            elif k > 0:
                coefs = [fac1, fac2]
        elif tau == 1:
            if k == 0:
                coefs = [1j]
            elif k > 0:
                coefs = [fac1 * 1j, -fac2 * 1j]
        return coefs, kval

    for j in range(MAX_J + 1):
        k_list = [k for k in range(-j, j + 1)]
        if linear:
            k = 0
            t = j % 2
            sym = C2V_KTAU_IRREPS[(k % 2, t)]
            jktau_list = [(j, k, t, sym)]
        else:
            jktau_list = []
            for k in range(0, j + 1):
                if k == 0:
                    tau = [j % 2]
                else:
                    tau = [0, 1]
                for t in tau:
                    sym = C2V_KTAU_IRREPS[(k % 2, t)]
                    jktau_list.append((j, k, t, sym))

        WANG_COEFS[(j, linear)] = (
            np.array(k_list),
            np.array(jktau_list),
            np.zeros((len(k_list), len(jktau_list)), dtype=np.complex128),
        )
        for i, (j, k, tau, sym) in enumerate(jktau_list):
            c, k_pair = _wang_coefs(j, k, tau)
            for kk, cc in zip(k_pair, c):
                i_k = k_list.index(kk)
                WANG_COEFS[(j, linear)][2][i_k, i] = cc


tabulate_wang(linear=True)
tabulate_wang(linear=False)


def _jplus(j, k, c=1):
    return (j, k - 1, jnp.sqrt(j * (j + 1) - k * (k - 1)) * c if abs(k - 1) <= j else 0)


def _jminus(j, k, c=1):
    return (j, k + 1, jnp.sqrt(j * (j + 1) - k * (k + 1)) * c if abs(k + 1) <= j else 0)


def _jx(j, k, c=1):
    return _sum_oper(
        [
            _jminus(j, k, c),
            _jplus(j, k, c),
        ],
        [0.5, 0.5],
    )


def _jy(j, k, c=1):
    return _sum_oper(
        [
            _jminus(j, k, c),
            _jplus(j, k, c),
        ],
        [0.5j, -0.5j],
    )


def _jz(j, k, c=1):
    return (j, k, k * c)


def _jj(j, k, c=1):
    return (j, k, j * (j + 1) * c)


def _jminus_jminus(j, k, c=1):
    return _jminus(*_jminus(j, k, c))


def _jminus_jplus(j, k, c=1):
    return _jminus(*_jplus(j, k, c))


def _jplus_jminus(j, k, c=1):
    return _jplus(*_jminus(j, k, c))


def _jplus_jplus(j, k, c=1):
    return _jplus(*_jplus(j, k, c))


def _jminus_jz(j, k, c=1):
    return _jminus(*_jz(j, k, c))


def _jplus_jz(j, k, c=1):
    return _jplus(*_jz(j, k, c))


def _jz_jminus(j, k, c=1):
    return _jz(*_jminus(j, k, c))


def _jz_jplus(j, k, c=1):
    return _jz(*_jplus(j, k, c))


def _sum_oper(oper: List[Tuple[int, int, float]], prefac: List[complex]):
    res = defaultdict(complex)
    for (j, k, c), fac in zip(oper, prefac):
        res[(j, k)] += c * fac
    return [(j, k, c) for (j, k), c in res.items()]


def _jx_jx(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [0.25, 0.25, 0.25, 0.25],
    )


def _jx_jy(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [0.25j, -0.25j, 0.25j, -0.25j],
    )


def _jx_jz(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jz(j, k, c),
            _jplus_jz(j, k, c),
        ],
        [0.5, 0.5],
    )


def _jy_jx(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [0.25j, 0.25j, -0.25j, -0.25j],
    )


def _jy_jy(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jminus(j, k, c),
            _jminus_jplus(j, k, c),
            _jplus_jminus(j, k, c),
            _jplus_jplus(j, k, c),
        ],
        [-0.25, 0.25, 0.25, -0.25],
    )


def _jy_jz(j, k, c=1):
    return _sum_oper(
        [
            _jminus_jz(j, k, c),
            _jplus_jz(j, k, c),
        ],
        [0.5j, -0.5j],
    )


def _jz_jx(j, k, c=1):
    return _sum_oper(
        [
            _jz_jminus(j, k, c),
            _jz_jplus(j, k, c),
        ],
        [0.5, 0.5],
    )


def _jz_jy(j, k, c=1):
    return _sum_oper(
        [
            _jz_jminus(j, k, c),
            _jz_jplus(j, k, c),
        ],
        [0.5j, -0.5j],
    )


def _jz_jz(j, k, c=1):
    return _jz(*_jz(j, k, c))


def _delta(x, y):
    return 1 if x == y else 0


def _overlap(jkc1, jkc2):
    if all(isinstance(elem, Iterable) for elem in jkc1):
        jkc1_ = jkc1
    else:
        jkc1_ = [jkc1]
    if all(isinstance(elem, Iterable) for elem in jkc2):
        jkc2_ = jkc2
    else:
        jkc2_ = [jkc2]
    return jnp.sum(
        jnp.array(
            [
                jnp.conj(c1) * c2 * _delta(j1, j2) * _delta(k1, k2)
                for (j1, k1, c1) in jkc1_
                for (j2, k2, c2) in jkc2_
            ]
        )
    )


def rotme_ovlp(j: int, linear: bool = False):
    k_list, jktau_list, wang_coefs = WANG_COEFS[(j, linear)]
    s = jnp.array(
        [[_overlap((j, k1, 1), (j, k2, 1)) for k2 in k_list] for k1 in k_list]
    )
    res = jnp.einsum("ki,kl,lj->ij", jnp.conj(wang_coefs), s, wang_coefs)
    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-12
    ), f"<J',k',tau'|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"
    return jnp.real(res), k_list, jktau_list


def rotme_rot(j: int, linear: bool = False):
    k_list, jktau_list, wang_coefs = WANG_COEFS[(j, linear)]
    jxx = jnp.array(
        [[_overlap((j, k1, 1), _jx_jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jxy = jnp.array(
        [[_overlap((j, k1, 1), _jx_jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jxz = jnp.array(
        [[_overlap((j, k1, 1), _jx_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jyx = jnp.array(
        [[_overlap((j, k1, 1), _jy_jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jyy = jnp.array(
        [[_overlap((j, k1, 1), _jy_jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jyz = jnp.array(
        [[_overlap((j, k1, 1), _jy_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jzx = jnp.array(
        [[_overlap((j, k1, 1), _jz_jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jzy = jnp.array(
        [[_overlap((j, k1, 1), _jz_jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jzz = jnp.array(
        [[_overlap((j, k1, 1), _jz_jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jmat = jnp.array([[jxx, jxy, jxz], [jyx, jyy, jyz], [jzx, jzy, jzz]])
    res = jnp.einsum("ki,abkl,lj->ijab", jnp.conj(wang_coefs), jmat, wang_coefs)
    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-12
    ), f"<J',k',tau'|Ja*Jb|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"
    return jnp.real(res), k_list, jktau_list


def rotme_cor(j: int, linear: bool = False):
    k_list, jktau_list, wang_coefs = WANG_COEFS[(j, linear)]
    jx = jnp.array(
        [[_overlap((j, k1, 1), _jx(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jy = jnp.array(
        [[_overlap((j, k1, 1), _jy(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jz = jnp.array(
        [[_overlap((j, k1, 1), _jz(j, k2)) for k2 in k_list] for k1 in k_list]
    )
    jvec = jnp.array([jx, jy, jz])
    res = 1j * jnp.einsum("ki,akl,lj->ija", jnp.conj(wang_coefs), jvec, wang_coefs)
    max_imag = jnp.max(jnp.abs(jnp.imag(res)))
    assert (
        max_imag < 1e-12
    ), f"i*<J',k',tau'|Ja|J,k,tau> matrix elements are not real-valued, max imaginary component: {max_imag}"
    return jnp.real(res), k_list, jktau_list


def symtop_on_grid(j: int, grid, linear: bool = False):
    k_list, jktau_list, wang_coefs = WANG_COEFS[(j, linear)]
    # psi[J+m, J+k, ipoint] for k,m = -J..J
    psi = np.sqrt((2 * j + 1) / (8 * np.pi**2)) * np.conj(wigner_D(j, grid))
    ind = [k for k in range(-j, j + 1)]
    map_ind = [ind.index(k) for k in k_list]
    res = jnp.einsum("ki,mkg->mig", wang_coefs, psi[:, map_ind, :])
    return res, k_list, jktau_list


def symtop_on_grid_split_angles(j: int, alpha, beta, gamma, linear: bool = False):
    k_list_wang, jktau_list, wang_coefs = WANG_COEFS[(j, linear)]
    ind = list(range(-j, j + 1))
    wang_map = [ind.index(k) for k in k_list_wang]
    ind = np.array(ind)

    v = _jy_eig(j)
    em = np.exp(1j * alpha[None, :] * ind[:, None])
    ek = np.exp(1j * gamma[None, :] * ind[:, None])

    res_k = np.einsum(
        "kt,ki,kg->tig", wang_coefs, v[wang_map], ek[wang_map], optimize="optimal"
    )  # (ktau, mu, point)

    res_m = np.einsum(
        "mi,mg->mig", np.conj(v), em, optimize="optimal"
    )  # (m, mu, point)

    eb = np.exp(-1j * beta[None, :] * ind[:, None])  # (mu, point)
    return res_k, res_m, eb, k_list_wang, jktau_list


def threej_wang(rank: int, j1: int, j2: int, linear: bool):
    """Computes three-j symbol contracted with the Cartesian-to-spherical
    transformation for tensor of specified rank, i.e.,

    \\sum_\\sigma=-\\omega^\\omega (-1)**k' threej(J, \\omega, J', k, \\sigma, -k') U_{\\omega,\\sigma,\\alpha}^{(\\omega)}

    and transformes the result into the Wang's basis.
    Here, \\omega = 0.. `rank` and \\alpha denotes Cartesian components of tensor,
    e.g., 'x', 'y', 'z' for rank-1 tensor, 'xx', 'xy', 'xz', 'yz', ..., 'zz' for rank-2 tensor.
    """
    k_list1, jktau_list1, wang_coefs1 = WANG_COEFS[(j1, linear)]
    k_list2, jktau_list2, wang_coefs2 = WANG_COEFS[(j2, linear)]

    k1 = np.array(k_list1)
    k2 = np.array(k_list2)
    k12 = np.concatenate(
        (
            k1[:, None, None].repeat(len(k2), axis=1),
            k2[None, :, None].repeat(len(k1), axis=0),
        ),
        axis=-1,
    ).reshape(-1, 2)
    n = len(k12)
    k12_1 = k12[:, 0]
    k12_2 = k12[:, 1]

    threej = {
        omega: np.zeros((2 * omega + 1, len(k1), len(k2)), dtype=np.complex128)
        for omega in UMAT_CART_TO_SPHER[rank].keys()
    }
    for omega, sigma in SPHER_IND[rank]:
        thrj = (-1) ** np.abs(k12_1) * wigner3j(
            [j2 * 2] * n,
            [omega * 2] * n,
            [j1 * 2] * n,
            k12_2 * 2,
            [sigma * 2] * n,
            -k12_1 * 2,
            ignore_invalid=True,
        )
        threej[omega][sigma + omega] = thrj.reshape(len(k1), len(k2))

    threej_wang = {}
    for omega in UMAT_CART_TO_SPHER[rank].keys():
        threej_wang[omega] = jnp.einsum(
            "ki,skl,lj,sc->ijc",
            jnp.conj(wang_coefs1),
            threej[omega],
            wang_coefs2,
            UMAT_CART_TO_SPHER[rank][omega],
        )
    return jktau_list1, jktau_list2, threej_wang