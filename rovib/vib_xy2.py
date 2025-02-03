"""Vibrational solutions for an XY2-type triatomic molecule"""

import itertools
from typing import Callable, List, Optional

import jax
import numpy as np
from jax import numpy as jnp
from numpy.typing import NDArray

jax.config.update("jax_enable_x64", True)


def vibrations_xy2(
    list_psi: List[Callable],
    list_x: List[NDArray[np.float64]],
    list_w: List[NDArray[np.float64]],
    list_q: List[NDArray[np.int_]],
    select_points: Callable,
    select_quanta: Callable,
    x_to_r_map: Callable,
    gmat: Callable,
    pseudo: Callable,
    potential: Callable,
    assign_c2v=False,
    ext_oper_list: Optional[Callable] = None,
):
    """Solves the eigenvalue problem for vibrational Hamiltonian of an XY2-type triatomic molecule

    See `h2s_rovib.ipynb` for the use example.
    """
    list_psi_vmap = [jax.jit(jax.vmap(psi, in_axes=(0, None))) for psi in list_psi]
    list_dpsi_vmap = [
        jax.jit(jax.vmap(jax.jacfwd(psi, argnums=0), in_axes=(0, None)))
        for psi in list_psi
    ]
    x_to_r_vmap = jax.jit(jax.vmap(x_to_r_map, in_axes=(0,)))
    jac_x_to_r_vmap = jax.jit(jax.vmap(jax.jacrev(x_to_r_map, argnums=0), in_axes=(0,)))

    # product quadrature grid

    x = jnp.stack([elem.ravel() for elem in jnp.meshgrid(*list_x)], axis=-1)
    w = jnp.prod(
        jnp.stack([elem.ravel() for elem in jnp.meshgrid(*list_w)], axis=-1), axis=-1
    )
    ind = jnp.where(jnp.array([select_points(x, w) for x, w in zip(x, w)]))
    x = x[ind]
    w = w[ind]
    x1, x2, x3 = x.T

    # product basis

    quanta = jnp.array(
        [elem for elem in itertools.product(*list_q) if select_quanta(elem)]
    )
    q1, q2, q3 = quanta.T

    psi1, psi2, psi3 = [
        f(x_, np.arange(0, np.max(n) + 1))[:, n]
        for f, x_, n in zip(list_psi_vmap, (x1, x2, x3), list_q)
    ]
    psi = psi1[:, q1] * psi2[:, q2] * psi3[:, q3]

    dpsi1, dpsi2, dpsi3 = [
        f(x_, np.arange(0, np.max(n) + 1))[:, n]
        for f, x_, n in zip(list_dpsi_vmap, (x1, x2, x3), list_q)
    ]

    dpsi = jnp.array(
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
    g = gmat(r)
    gvib = g[:, :3, :3]  # vibrational part
    grot = g[:, 3:6, 3:6]  # rotational part
    gcor = g[:, :3, 3:6]  # Coriolis part
    vmat = jnp.einsum("gi,gj,g,g->ij", psi, psi, potential(r), w)
    umat = jnp.einsum("gi,gj,g,g->ij", psi, psi, pseudo(r), w)
    dpsi_r = jnp.einsum("xgi,gxy->giy", dpsi, inv_jac_r)
    tmat = jnp.einsum("gix,gjy,gxy,g->ij", dpsi_r, dpsi_r, gvib, w)

    # eigenvectors and eigenvalues

    hmat = 0.5 * tmat + vmat + umat
    enr, vec = jnp.linalg.eigh(hmat)
    psi = jnp.einsum("gi,ij->gj", psi, vec)
    dpsi = jnp.einsum("xgi,ij->xgj", dpsi, vec)
    dpsi_r = jnp.einsum("gix,ij->gjx", dpsi_r, vec)

    # symmetry labels of eigenstates

    sym = ["X" for _ in range(len(quanta))]  # default

    if assign_c2v:
        sym = []

        # P(12) applied to eigenfunctions on grid
        psi1, psi2, psi3 = [
            f(x_, np.arange(0, np.max(n) + 1))[:, n]
            for f, x_, n in zip(list_psi_vmap, (x2, x1, x3), list_q)
        ]
        p12_psi = psi1[:, q1] * psi2[:, q2] * psi3[:, q3]
        p12_psi = jnp.einsum("gi,ij->gj", p12_psi, vec)

        # check symmetry relation
        for i in range(psi.shape[-1]):
            ind = np.where(np.abs(psi[:, i]) > 1e-06)
            ratio = np.mean(psi[ind, i] / p12_psi[ind, i])
            if np.abs(ratio - 1) < 1e-6:
                sym.append("A1")
            elif np.abs(ratio + 1) < 1e-6:
                sym.append("B2")
            else:
                raise ValueError(
                    f"Can't determine symmetry, the ratio psi / P(12)psi = {ratio} "
                    + f"for the state with energy = {enr[i]} is nether close to 1 nor -1"
                )

    # matrix elements of rotational operators

    grot_me = jnp.einsum("gi,gj,gab,g->ijab", psi, psi, grot, w)

    gcor_me = jnp.einsum("gip,gj,gpa,g->ija", dpsi_r, psi, gcor, w) - jnp.einsum(
        "gi,gjp,gpa,g->ija", psi, dpsi_r, gcor, w
    )

    ext_oper_me = []
    if ext_oper_list is not None:
        for oper in ext_oper_list:
            me = jnp.einsum("gi,gj,g...,g->ij...", psi, psi, oper(r), w)
            ext_oper_me.append(me)

    return enr, vec, sym, quanta, grot_me, gcor_me, ext_oper_me