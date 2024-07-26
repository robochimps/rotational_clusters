from jax import config
from jax import numpy as jnp

from .h2s_potential import poten
from .keo_xy2 import gmat1_sinrho, gmat2_sinrho, pseudo_sinrho
from .molecule import Molecule

config.update("jax_enable_x64", True)


def _poten(coords):
    r1, r2, rho = coords.T
    alpha = jnp.pi - rho
    return poten(jnp.array([r1, r2, alpha]).T) * jnp.sin(rho)


def _overlap(coords):
    r1, r2, rho = coords.T
    return jnp.sin(rho)


H2S = Molecule(
    masses=jnp.array([31.97207070, 1.00782505, 1.00782505]),
    potential=_poten,
    gmat1=gmat1_sinrho,
    gmat2=gmat2_sinrho,
    pseudo=pseudo_sinrho,
    overlap=_overlap,
)