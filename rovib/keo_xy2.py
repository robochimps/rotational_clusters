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


def _gmat1_sinrho(coords, mass_x, mass_y):
    r1, r2, rho = coords
    g11 = jnp.sin(rho) * (1.0 / (2 * mass_x) + 1.0 / (2 * mass_y))
    g12 = -jnp.sin(rho) * jnp.cos(rho) / (2.0 * mass_x)
    g13 = jnp.sin(rho) ** 2 / (2.0 * r2 * mass_x)
    g22 = jnp.sin(rho) * (1.0 / (2.0 * mass_x) + 1.0 / (2.0 * mass_y))
    g23 = jnp.sin(rho) ** 2 / (2.0 * r1 * mass_x)
    g33 = jnp.sin(rho) * (
        1.0 / (2.0 * r1**2 * mass_x)
        + 1.0 / (2.0 * r2**2 * mass_x)
        + 1.0 / (2.0 * r1**2 * mass_y)
        + 1.0 / (2.0 * r2**2 * mass_y)
        + jnp.cos(rho) / (r1 * r2 * mass_x)
    )
    return jnp.array([[g11, g12, g13], [g12, g22, g23], [g13, g23, g33]]) * KEO_INVCM


gmat1_sinrho = jax.jit(jax.vmap(_gmat1_sinrho, in_axes=(0, None, None)))


def _gmat2_sinrho(coords, mass_x, mass_y):
    r1, r2, rho = coords
    g1 = jnp.sin(rho) * jnp.cos(rho) / (4 * r2 * mass_x)
    g2 = jnp.sin(rho) * jnp.cos(rho) / (4 * r1 * mass_x)
    g3 = jnp.cos(rho) * (
        1.0 / (4.0 * r1**2 * mass_x)
        + 1.0 / (4.0 * r2**2 * mass_x)
        + 1.0 / (4.0 * r1**2 * mass_y)
        + 1.0 / (4.0 * r2**2 * mass_y)
        + jnp.cos(rho) / (2.0 * r1 * r2 * mass_x)
    )
    return jnp.array([g1, g2, g3]) * KEO_INVCM


gmat2_sinrho = jax.jit(jax.vmap(_gmat2_sinrho, in_axes=(0, None, None)))


def _pseudo_sinrho(coords, mass_x, mass_y):
    r1, r2, rho = coords
    return (
        -jnp.sin(rho)
        * (
            1.0 / (4.0 * r1**2 * mass_x)
            + 1.0 / (4.0 * r2**2 * mass_x)
            + 1.0 / (4.0 * r1**2 * mass_y)
            + 1.0 / (4.0 * r2**2 * mass_y)
        )
        * KEO_INVCM
    )


pseudo_sinrho = jax.jit(jax.vmap(_pseudo_sinrho, in_axes=(0, None, None)))