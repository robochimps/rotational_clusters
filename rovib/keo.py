import functools

import jax
from jax import config
from jax import numpy as jnp
from scipy import constants

config.update("jax_enable_x64", True)

KEO_INVCM = (
    constants.value("Planck constant")
    * constants.value("Avogadro constant")
    * 1e16
    / (4.0 * jnp.pi**2 * constants.value("speed of light in vacuum"))
    * 1e5
)

eps = jnp.array(
    [
        [[int((i - j) * (j - k) * (k - i) * 0.5) for k in range(3)] for j in range(3)]
        for i in range(3)
    ],
    dtype=jnp.float64,
)


class Molecule_meta(type):
    def __init__(cls, *args, **kwargs):
        cls._masses = None

    @classmethod
    def internal_to_cartesian(cls, q):
        raise Exception(
            "Please specify internal-to-Cartesian transformation function:\n"
            + "\t'Molecule.internal_to_cartesian = function(q: jax.np.array(no. coords)) -> jax.np.array(no. atoms, 3)'"
        )

    @classmethod
    def cartesian_to_internal(cls, xyz):
        raise Exception(
            "Please specify internal-to-Cartesian transformation function:\n"
            + "\t'Molecule.internal_to_cartesian = function(q: jax.np.array(no. coords)) -> jax.np.array(no. atoms, 3)'"
        )

    @property
    def masses(cls):
        if cls._masses is None:
            raise Exception(
                f"Please specify atomic masses:\n\t'Molecule.masses: np.array(no. atoms)'"
            )
        return cls._masses

    @masses.setter
    def masses(cls, val):
        try:
            val.shape
        except AttributeError:
            raise Exception(f"Atomic masses must be a numpy array") from None
        cls._masses = val


class Molecule(metaclass=Molecule_meta):
    pass


def com(cart):
    @functools.wraps(cart)
    def wrapper_com(*args, **kwargs):
        xyz = cart(*args, **kwargs)
        masses = Molecule.masses
        com = jnp.dot(masses, xyz) / jnp.sum(masses)
        return xyz - com[None, :]

    return wrapper_com


@jax.jit
def gmat(q):
    xyz_g = jax.jacfwd(Molecule.internal_to_cartesian)(q)
    tvib = xyz_g
    xyz = Molecule.internal_to_cartesian(q)
    natoms = xyz.shape[0]
    trot = jnp.transpose(jnp.dot(eps, xyz.T), (2, 0, 1))
    ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(natoms)])
    tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
    masses_sq = jnp.array([jnp.sqrt(Molecule.masses[i]) for i in range(natoms)])
    tvec = tvec * masses_sq[:, jnp.newaxis, jnp.newaxis]
    tvec = jnp.reshape(tvec, (natoms * 3, len(q) + 6))
    return jnp.dot(tvec.T, tvec)


@jax.jit
def Gmat(q):
    return jnp.linalg.inv(gmat(q)) * KEO_INVCM


batch_Gmat = jax.jit(jax.vmap(Gmat, in_axes=0))


@jax.jit
def dGmat(q):
    return jax.jacfwd(Gmat)(q)


batch_dGmat = jax.jit(jax.vmap(dGmat, in_axes=0))


@jax.jit
def Detgmat(q):
    nq = len(q)
    return jnp.linalg.det(gmat(q)[: nq + 3, : nq + 3])


@jax.jit
def dDetgmat(q):
    return jax.grad(Detgmat)(q)


@jax.jit
def hDetgmat(q):
    # return jax.jacfwd(jax.jacrev(Detgmat))(q)
    return jax.jacfwd(jax.jacfwd(Detgmat))(q)


@jax.jit
def pseudo(q):
    nq = len(q)
    G = Gmat(q)[:nq, :nq]
    dG = dGmat(q)[:nq, :nq, :]
    dG = jnp.transpose(dG, (0, 2, 1))
    det = Detgmat(q)
    det2 = det * det
    ddet = dDetgmat(q)
    hdet = hDetgmat(q)
    pseudo1 = (jnp.dot(ddet, jnp.dot(G, ddet))) / det2
    pseudo2 = (jnp.sum(jnp.diag(jnp.dot(dG, ddet))) + jnp.sum(G * hdet)) / det
    return (-3 * pseudo1 + 4 * pseudo2) / 32.0


batch_pseudo = jax.jit(jax.vmap(pseudo, in_axes=0))