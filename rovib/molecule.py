from dataclasses import dataclass
from typing import Callable, Optional

import jax
import numpy as np
from jax import config
from jax import numpy as jnp
from numpy.typing import NDArray
from scipy import optimize

config.update("jax_enable_x64", True)


@dataclass
class Molecule:
    masses: NDArray[np.float_]
    potential: Callable
    gmat1: Callable
    gmat2: Callable
    pseudo: Callable
    overlap: Optional[Callable] = lambda x: jnp.ones(x.shape[0])

    def linear_mapping_ho(self):
        vmin = optimize.minimize(self.potential, [1.0, 1.0, np.pi / 2])
        r0 = vmin.x
        v0 = vmin.fun
        freq = np.diag(jax.hessian(self.potential)(r0))  # NOTE multiply by 2?
        mu = np.diag(self.gmat1(np.array([r0]), self.masses[0], self.masses[1])[0])
        a = np.sqrt(np.sqrt(mu / freq))
        b = r0
        return a, b, r0, v0