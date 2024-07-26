import itertools
from typing import Callable, Tuple

import jax
import numpy as np
from jax import numpy as jnp
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from .basis import hermite, legendre
from .h2s import H2S


def solve(
    list_psi: Tuple[Callable, Callable, Callable],
    list_x: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
    list_w: Tuple[NDArray[np.float_], NDArray[np.float_], NDArray[np.float_]],
    list_q: Tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]],
    select_points: Callable,
    select_quanta: Callable,
    x_to_r_map: Callable,
    gmat1: Callable,
    gmat2: Callable,
    pseudo: Callable,
    potential: Callable,
    overlap_func: Callable,
):

    list_psi_vmap = [jax.jit(jax.vmap(psi, in_axes=(0, None))) for psi in list_psi]
    list_dpsi_vmap = [
        jax.jit(jax.vmap(jax.jacrev(bas, argnums=0), in_axes=(0, None)))
        for bas in list_psi
    ]
    x_to_r_vmap = jax.jit(jax.vmap(x_to_r_map, in_axes=(0,)))
    jac_x_to_r_vmap = jax.jit(jax.vmap(jax.jacrev(x_to_r_map, argnums=0), in_axes=(0,)))

    # quadrature product grid

    x = np.stack([elem.ravel() for elem in np.meshgrid(*list_x)], axis=-1)
    w = np.prod(
        np.stack([elem.ravel() for elem in np.meshgrid(*list_w)], axis=-1), axis=-1
    )
    ind = np.where([select_points(x, w) for x, w in zip(x, w)])
    x = x[ind]
    w = w[ind]

    # product basis

    quanta = np.array([elem for elem in itertools.product(*list_q) if select_quanta(elem)])
    q1, q2, q3 = quanta.T

    psi1, psi2, psi3 = [
        f(x_, np.arange(0, np.max(n) + 1))[:, n]
        for f, x_, n in zip(list_psi_vmap, x.T, list_q)
    ]

    dpsi1, dpsi2, dpsi3 = [
        f(x_, np.arange(0, np.max(n) + 1))[:, n]
        for f, x_, n in zip(list_dpsi_vmap, x.T, list_q)
    ]

    psi = psi1[:, q1] * psi2[:, q2] * psi3[:, q3]
    dpsi = np.array(
        [
            dpsi1[:, q1] * psi2[:, q2] * psi3[:, q3],
            psi1[:, q1] * dpsi2[:, q2] * psi3[:, q3],
            psi1[:, q1] * psi2[:, q2] * dpsi3[:, q3],
        ]
    )

    # operators on quadrature grid

    r = x_to_r_vmap(x)
    jac_r = jac_x_to_r_vmap(x)
    inv_jac_r = jnp.linalg.inv(jac_r)
    print(np.min(x, axis=0), np.max(x, axis=0))
    print(np.min(r, axis=0), np.max(r, axis=0))

    print(gmat1(r).shape, gmat2(r).shape)
    pot = jnp.einsum("gi,gj,g,g->ij", psi, psi, potential(r), w)
    pseudopot = jnp.einsum("gi,gj,g,g->ij", psi, psi, pseudo(r), w)
    ovlp = jnp.einsum("gi,gj,g,g->ij", psi, psi, overlap_func(r), w)
    dpsi_ = jnp.einsum("xgi,gxy->giy", dpsi, inv_jac_r)
    keo1 = jnp.einsum("gix,gjy,gxy,g->ij", dpsi_, dpsi_, gmat1(r), w)
    keo2 = jnp.einsum("gix,gx,gj,g->ij", dpsi_, gmat2(r), psi, w)

    s_diag, s_vec = jnp.linalg.eigh(ovlp)
    s_invsqrt = s_vec @ jnp.diag(1.0 / jnp.sqrt(s_diag)) @ s_vec.T
    h = s_invsqrt @ (keo1 + keo2 + pot + pseudopot) @ s_invsqrt
    e, v = jnp.linalg.eigh(h)
    print(e[0], e[:10] - e[0])


if __name__ == "__main__":

    lin_a, lin_b, *_ = H2S.linear_mapping_ho()

    x_to_r_map = lambda x: jnp.array(
        [
            x[0] * lin_a[0] + lin_b[0],
            x[1] * lin_a[1] + lin_b[1],
            x[2] * lin_a[2] + lin_b[2],
            # jnp.arccos(x[2]),
        ]
    )

    n1, n2, n3 = (60, 60, 60)
    x1, w1 = hermgauss(n1)
    x2, w2 = hermgauss(n2)
    x3, w3 = hermgauss(n3)
    # x3, w3 = leggauss(n3)
    w1 /= np.exp(-(x1**2))
    w2 /= np.exp(-(x2**2))
    w3 /= np.exp(-(x3**2))

    # list_psi = (hermite, hermite, legendre)
    list_psi = (hermite, hermite, hermite)
    list_x = (x1, x2, x3)
    list_w = (w1, w2, w3)
    list_q = (np.arange(40), np.arange(40), np.arange(40))
    solve(
        list_psi,
        list_x,
        list_w,
        list_q,
        select_points=lambda x, w: True,
        select_quanta=lambda q: np.sum(q * np.array([2, 2, 1])) <= 18,
        x_to_r_map=x_to_r_map,
        gmat1=H2S.gmat1,
        gmat2=H2S.gmat2,
        pseudo=H2S.pseudo,
        potential=H2S.potential,
        overlap_func=H2S.overlap,
    )
    pass