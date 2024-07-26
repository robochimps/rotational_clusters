from scipy.special import factorial
from jax import config
from jax import Array
import jax
import numpy as np
from numpy.polynomial.hermite import hermval, hermder
from numpy.polynomial.legendre import legval, legder
from numpy.typing import NDArray

config.update("jax_enable_x64", True)


def _hermite(x, n):
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag(1.0 / np.sqrt(2.0**n * factorial(n)) / sqsqpi)
    f = hermval(np.asarray(x), c) * np.exp(-(x**2) / 2)
    return f.T


def _hermite_deriv(x, n):
    sqsqpi = np.sqrt(np.sqrt(np.pi))
    c = np.diag(1.0 / np.sqrt(2.0**n * factorial(n)) / sqsqpi)
    h = hermval(np.asarray(x), c)
    dh = hermval(np.asarray(x), hermder(c, m=1))
    f = (dh - h * x) * np.exp(-(x**2) / 2)
    return f.T


def _legendre(x, n):
    coef = np.diag(1.0 / np.sqrt(2 / (2 * n + 1)))
    f = legval(np.asarray(x), coef)
    return f.T


def _legendre_deriv(x, n):
    coef = np.diag(1.0 / np.sqrt(2 / (2 * n + 1)))
    df = legval(np.asarray(x), legder(coef, m=1))
    return df.T


@jax.custom_jvp
def hermite(x: NDArray[np.float_], n: NDArray[np.int_]) -> Array:
    shape_dtype = jax.ShapeDtypeStruct(x.shape + n.shape, x.dtype)
    return jax.pure_callback(_hermite, shape_dtype, x, n, vectorized=True)


def _jvp_hermite_deriv(x, n):
    shape_dtype = jax.ShapeDtypeStruct(x.shape + n.shape, x.dtype)
    return jax.pure_callback(_hermite_deriv, shape_dtype, x, n, vectorized=True)


@hermite.defjvp
def _hermite_jvp(prim, tang):
    x, n = prim
    x_dot, n_dot = tang
    prim_out = hermite(x, n)
    tang_out = _jvp_hermite_deriv(x, n) * x_dot
    return prim_out, tang_out


@jax.custom_jvp
def legendre(x: NDArray[np.float_], n: NDArray[np.int_]) -> Array:
    shape_dtype = jax.ShapeDtypeStruct(x.shape + n.shape, x.dtype)
    return jax.pure_callback(_legendre, shape_dtype, x, n, vectorized=True)


def _jvp_legendre_deriv(x, n):
    shape_dtype = jax.ShapeDtypeStruct(x.shape + n.shape, x.dtype)
    return jax.pure_callback(_legendre_deriv, shape_dtype, x, n, vectorized=True)


@legendre.defjvp
def _legendre_jvp(prim, tang):
    x, n = prim
    x_dot, n_dot = tang
    prim_out = legendre(x, n)
    tang_out = _jvp_legendre_deriv(x, n) * x_dot
    return prim_out, tang_out