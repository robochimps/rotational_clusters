import numpy as np
from jax import config
from jax import numpy as jnp

from .h2s_potential import poten
from .keo_xy2 import gmat1_sinrho, gmat2_sinrho, pseudo_sinrho
from .molecule import Molecule

config.update("jax_enable_x64", True)

MASSES = np.array([31.97207070, 1.00782505, 1.00782505])


def _poten(coords):
    r1, r2, rho = coords.T
    alpha = jnp.pi - rho
    return poten(jnp.array([r1, r2, alpha]).T) * jnp.sin(rho)


def _overlap(coords):
    r1, r2, rho = coords.T
    return jnp.sin(rho)


def _gmat1(coords):
    return gmat1_sinrho(coords, MASSES[0], MASSES[1])


def _gmat2(coords):
    return gmat2_sinrho(coords, MASSES[0], MASSES[1])


def _pseudo(coords):
    return pseudo_sinrho(coords, MASSES[0], MASSES[1])


H2S = Molecule(
    masses=MASSES,
    potential=_poten,
    gmat1=_gmat1,
    gmat2=_gmat2,
    pseudo=_pseudo,
    overlap=_overlap,
)