import jax
import numpy as np
from jax import config
from jax import numpy as jnp
from numpy.polynomial.hermite import hermgauss
from scipy import optimize

from .h2s_potential import potential
from .keo import Molecule, batch_Gmat, batch_pseudo, com
from .primbas import hermite
from .symtop import rotme_cor, rotme_ovlp, rotme_rot
from .vib_xy2 import vibrations_xy2
from .c2v import C2V_PRODUCT_TABLE

config.update("jax_enable_x64", True)


MASS_S = 31.97207070
MASS_H = 1.00782505


@com
def valence_bond_coordinates(coords):
    r1, r2, angle = coords
    return jnp.array(
        [
            [0.0, 0.0, 0.0],
            [r1 * jnp.cos(angle / 2), 0.0, r1 * jnp.sin(angle / 2)],
            [r2 * jnp.cos(angle / 2), 0.0, -r2 * jnp.sin(angle / 2)],
        ]
    )


Molecule.masses = np.array([MASS_S, MASS_H, MASS_H])
Molecule.internal_to_cartesian = valence_bond_coordinates


if __name__ == "__main__":

    # init rotational matrix elements
    j_angmom = 9
    linear = False
    s_rot, k_list, ktau_list = rotme_ovlp(j_angmom, linear)
    jab_rot, k_list, ktau_list = rotme_rot(j_angmom, linear)
    ja_rot, k_list, ktau_list = rotme_cor(j_angmom, linear)
    nbas_rot = len(k_list)
    print("rotational angular momentum:", j_angmom)
    print("number of rotational functions:", nbas_rot)

    NCOO = 3

    def grot(x):
        g = batch_Gmat(x)
        return jnp.einsum(
            "abij,gab->gij", jab_rot, g[:, NCOO : NCOO + 3, NCOO : NCOO + 3]
        )

    def gcor(x):
        g = batch_Gmat(x)
        return jnp.einsum("aij,gpa->gpij", ja_rot, g[:, :NCOO, NCOO : NCOO + 3])

    vmin = optimize.minimize(potential, [1.0, 1.0, np.pi / 2])
    r0 = vmin.x
    v0 = vmin.fun
    freq = np.diag(jax.hessian(potential)(r0))  # NOTE multiply by 2?
    mu = np.diag(batch_Gmat(jnp.array([r0]))[0, :3, :3])
    a = np.sqrt(np.sqrt(mu / freq))
    b = r0
    print(a, b)
    x_to_r_map = lambda x: a * x + b

    # list of 1D primitive basis functions for each vibrational coordinate
    list_psi = (hermite, hermite, hermite)

    # 1D contracted basis sets

    pmax = 10
    nmax = 20
    nmax0 = 0
    npoints = 30
    npoints0 = 10

    enr = []
    vec = []
    for icoo in range(3):
        # quanta
        list_q = [[nmax0]] * 3
        list_q[icoo] = np.arange(nmax)

        # quadratures
        n = [npoints0] * 3
        n[icoo] = npoints
        n1, n2, n3 = n
        x1, w1 = hermgauss(n1)
        x2, w2 = hermgauss(n2)
        x3, w3 = hermgauss(n3)
        w1 /= np.exp(-(x1**2))
        w2 /= np.exp(-(x2**2))
        w3 /= np.exp(-(x3**2))
        list_x = (x1, x2, x3)
        list_w = (w1, w2, w3)

        # solutions
        e, v, *_ = vibrations_xy2(
            list_psi,
            list_x,
            list_w,
            list_q,
            select_points=lambda x, w: True,
            select_quanta=lambda q: np.sum(q * np.array([1, 1, 1])) <= nmax,
            x_to_r_map=x_to_r_map,
            gmat=lambda x: batch_Gmat(x)[:, :3, :3],
            pseudo=batch_pseudo,
            potential=potential,
        )

        # keep eigenvectors
        enr.append(e)
        vec.append(v)

        print(f"Energies for coordinate #{icoo}:")
        print(e[0], e - e[0])

    # primitive to contracted basis
    list_psi = [
        lambda x, n: jnp.dot(hermite(x, n), vec[0]),
        lambda x, n: jnp.dot(hermite(x, n), vec[1]),
        lambda x, n: jnp.dot(hermite(x, n), vec[2]),
    ]

    # quanta
    list_q = [np.arange(nmax)] * 3

    # quadratures
    n = [npoints] * 3
    n1, n2, n3 = n
    x1, w1 = hermgauss(n1)
    x2, w2 = hermgauss(n2)
    x3, w3 = hermgauss(n3)
    w1 /= np.exp(-(x1**2))
    w2 /= np.exp(-(x2**2))
    w3 /= np.exp(-(x3**2))
    list_x = (x1, x2, x3)
    list_w = (w1, w2, w3)

    # solutions
    e, v, quanta, rot_me, cor_me = vibrations_xy2(
        list_psi,
        list_x,
        list_w,
        list_q,
        select_points=lambda x, w: True,
        select_quanta=lambda q: np.sum(q * np.array([2, 2, 1])) <= pmax,
        x_to_r_map=x_to_r_map,
        gmat=lambda x: batch_Gmat(x)[:, :3, :3],
        pseudo=batch_pseudo,
        potential=potential,
        grot=grot,
        gcor=gcor,
        assign_c2v=True,
    )
    print(quanta)
    print(e[0], e - e[0])
    print(rot_me.shape, cor_me.shape)

    rot_sym = ktau_list[:, -1]
    vib_sym = quanta[:, -1]
    tot_sym = np.concatenate(
        (
            vib_sym[:, None].repeat(len(rot_sym), axis=1),
            rot_sym[None, :].repeat(len(vib_sym), axis=0),
        ),
        axis=-1,
    ).reshape(-1, 2)
    print(tot_sym)
    prod_sym = np.array([C2V_PRODUCT_TABLE[(sv, sr)] for (sv, sr) in tot_sym])
    tot_sym = np.concatenate((tot_sym, prod_sym[:, None]), axis=-1)

    ind = np.where(tot_sym[:, -1] == "A1")
    print(tot_sym[ind])
    exit()

    h = (
        jnp.diag(e)[:, :, None, None] * s_rot[None, None, :, :]
        + 0.5 * rot_me
        + 0.5 * cor_me
    )
    print(h.shape)
    nvib = h.shape[0]
    nrot = h.shape[2]
    h = h.swapaxes(1, 2).reshape(nvib * nrot, -1)
    print(h.shape)
    e, v = jnp.linalg.eigh(h)
    zpe = 3291.1604451003955
    print(e[:21] - zpe)

    q = np.concatenate(
        (
            quanta[:, None, :].repeat(nrot, axis=1),
            ktau_list[None, :, :].repeat(nvib, axis=0),
        ),
        axis=-1,
    ).reshape(-1, 7)

    ind = np.argmax(v**2, axis=0)
    for q_, i in zip(q[ind], range(21)):
        print(e[i] - zpe, q_)