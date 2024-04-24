from functools import partial
import jax
from jax import numpy as jnp
import numpy as np
from typing import Optional


# Default quadrotor parameters
m_q = 1.0 # kg
I_xx = 0.1 # kg.m^2
l_q = 0.3 # m, length of the quadrotor
g = 9.81


class PlanarQuadrotorEnv:
    def __init__(self, config: dict = None, state : Optional[jnp.ndarray]=None):
        if config is None:
            self.m_q = m_q
            self.I_xx = I_xx
            self.g = g
            self.l_q = l_q
        else:
            self.m_q = config["simulator"]["m_q"]
            self.I_xx = config["simulator"]["I_xx"]
            self.g = config["simulator"]["g"]
            self.l_q = config["simulator"]["l_q"]

        self.state: Optional[jnp.ndarray] = state
            
    @partial(jax.jit, static_argnums=0)
    def step(self, state=None, control=[0, 0], dt: float=0.01):
        """
        dynamics with JAX-compatible code.
        """
        if state is None:
            state = self.state
            if state is None: raise Exception("state variable is not defined.")
        
        y, y_dot, z, z_dot, phi, phi_dot = state
        u1, u2 = control
        # Quadrotor dynamics
        y_ddot = -u1 * jnp.sin(phi) / self.m_q
        z_ddot = -self.g + u1 * jnp.cos(phi) / self.m_q
        phi_ddot = u2 / self.I_xx
    
        next_state = state + jnp.array([y_dot+y_ddot*dt, y_ddot, z_dot+z_ddot*dt, z_ddot, phi_dot+phi_ddot*dt, phi_ddot]) * dt
        self.state = next_state
        return next_state
