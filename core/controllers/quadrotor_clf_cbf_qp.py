from typing import Dict, List
import numpy as np
import osqp
from scipy import sparse

from core.controllers.base_controller import BaseController


class QuadrotorCLFCBFController(BaseController):
    """
    A CLF-CBF safety filter assuming a simple velocity-controled dynamics
        y_dot = u1
        z_dot = u2
    Barrier funciton h is defined as the distances to each obstacle
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(device)
        self.obstacle_info = {"center": [], "radius": []}
        self.set_config(config)

    def predict_action(self, obs_dict: Dict[str, List], control: np.ndarray, target_position: np.ndarray) -> np.ndarray:
        for center, radius in zip(obs_dict["obstacle_info"]["center"], obs_dict["obstacle_info"]["radius"]):
            self.set_obstacle(center, radius)

        safe_command = self.clf_cbf_control(
            state=obs_dict["state"],
            control=control,
            obs_center=self.obstacle_info["center"],
            obs_radius=self.obstacle_info["radius"],
            cbf_alpha=self.cbf_alpha,
            clf_gamma=self.clf_gamma,
            penalty_slack_cbf=self.penalty_slack_cbf,
            penalty_slack_clf=self.penalty_slack_clf,
            target_position=target_position,
        )
        return safe_command

    def set_obstacle(self, center: tuple, radius: float):
        self.obstacle_info = {"center": [], "radius": []}
        self.obstacle_info["center"].append(center)
        self.obstacle_info["radius"].append(radius)

    def set_config(self, config: Dict):
        self.cbf_alpha = config["cbf_clf_controller"]["cbf_alpha"]
        self.clf_gamma = config["cbf_clf_controller"]["clf_gamma"]
        self.penalty_slack_cbf = config["cbf_clf_controller"]["penalty_slack_cbf"]
        self.penalty_slack_clf = config["cbf_clf_controller"]["penalty_slack_clf"]
        self.denoising_guidance_step = config["cbf_clf_controller"]["denoising_guidance_step"]
        self.quadrotor_params = config["simulator"]

    @staticmethod
    def _barrier_func(y, z, obs_y, obs_z, obs_r) -> float:
        return (y - obs_y) ** 2 + (z - obs_z) ** 2 - (obs_r) ** 2

    @staticmethod
    def _barrier_func_dot(y, z, obs_y, obs_z) -> list:
        return [2 * (y - obs_y), 2 * (z - obs_z)]

    @staticmethod
    def _lyapunoc_func(y, z, des_y, des_z) -> float:
        return (y - des_y) ** 2 + (z - des_z) ** 2

    @staticmethod
    def _lyapunov_func_dot(y, z, des_y, des_z) -> list:
        return [2 * (y - des_y), 2 * (z - des_z)]

    @staticmethod
    def _define_QP_problem_data(
        u1: float,
        u2: float,
        cbf_alpha: float,
        clf_gamma: float,
        penalty_slack_cbf: float,
        penalty_slack_clf: float,
        h: list,
        coeffs_dhdx: list,
        v: list,
        coeffs_dvdx: list,
        vmin=-15.0,
        vmax=15.0,
    ):
        vmin, vmax = -15.0, 15.0

        P = sparse.csc_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, penalty_slack_cbf, 0], [0, 0, 0, penalty_slack_clf]])
        q = np.array([-u1, -u2, 0, 0])
        A = sparse.csc_matrix(
            [c for c in coeffs_dhdx]
            + [c for c in coeffs_dvdx]
            + [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        lb = np.array([-cbf_alpha * h_ for h_ in h] + [-np.inf for _ in v] + [vmin, vmin, 0, 0])
        ub = np.array([np.inf for _ in h] + [-clf_gamma * v_ for v_ in v] + [vmax, vmax, np.inf, np.inf])
        return P, q, A, lb, ub

    @staticmethod
    def _get_quadrotor_state(state):
        y, y_dot, z, z_dot, phi, phi_dot = state
        return y, y_dot, z, z_dot, phi, phi_dot

    def _calculate_cbf_coeffs(self, state: np.ndarray, obs_center: List, obs_radius: List, minimal_distance: float):
        """
        Let barrier function be h and system state x, the CBF constraint
        h_dot(x) >= - alpha * h + δ
        """
        h = []  # barrier values (here, remaining distance to each obstacle)
        coeffs_dhdx = []  # dhdt = dhdx * dxdt = dhdx * u
        for center, radius in zip(obs_center, obs_radius):
            y, _, z, _, _, _ = self._get_quadrotor_state(state)
            h.append(self._barrier_func(y, z, center[0], center[1], radius + minimal_distance))
            # Additional [1, 0] incorporates the CBF slack variable into the constraint
            coeffs_dhdx.append(self._barrier_func_dot(y, z, center[0], center[1]) + [1, 0])
        return h, coeffs_dhdx

    def _calculate_clf_coeffs(self, state: np.ndarray, target_y: float, _target_z: float):
        """
        Let Lyapunov function be v and system state x, the CBF constraint
        v_dot(x) - δ <= - gamma * v
        """
        y, _, z, _, _, _ = self._get_quadrotor_state(state)
        v = [self._lyapunoc_func(y, z, target_y, _target_z)]
        # Additional [0, -1] incorporates the CLF slack variable into the constraint
        coeffs_dvdx = [self._lyapunov_func_dot(y, z, target_y, _target_z) + [0, -1]]
        return v, coeffs_dvdx

    def clf_cbf_control(
        self,
        state: np.ndarray,
        control: np.ndarray,
        obs_center: List,
        obs_radius: List,
        cbf_alpha: float = 15.0,
        clf_gamma: float = 0.01,
        penalty_slack_cbf: float = 1e2,
        penalty_slack_clf: float = 1.0,
        target_position: tuple = (5.0, 5.0),
    ):
        """
        Calculate the safe command by solveing the following optimization problem

                    minimize  || u - u_nom ||^2 + k * δ^2
                      u, δ
                    s.t.
                            h'(x) ≥ -𝛼 * h(x) - δ1
                            v'(x) ≤ -γ * v(x) + δ2
                            u_min ≤ u ≤ u_max
                                0 ≤ δ1,δ2 ≤ inf
        where
            u = [ux, uy] is the control input in x and y axis respectively.
            δ is the slack variable
            h(x) is the control barrier function and h'(x) its derivative
            v(x) is the lyapunov function and v'(x) its derivative

        The problem above can be formulated as QP (ref: https://osqp.org/docs/solver/index.html)

                    minimize 1/2 * x^T * Px + q^T x
                        x
                    s.t.
                                l ≤ Ax ≤ u
        where
            x = [ux, uy, δ1, δ2]

        """
        u1, u2 = control
        target_y, target_z = target_position

        # Calculate values of the barrier function and coeffs in h_dot to state
        h, coeffs_dhdx = self._calculate_cbf_coeffs(state, obs_center, obs_radius, self.quadrotor_params["l_q"])
        # Calculate value of the lyapunov function and coeffs in v_dot to state
        v, coeffs_dvdx = self._calculate_clf_coeffs(state, target_y, target_z)

        # Define problem
        P, q, A, lb, ub = self._define_QP_problem_data(
            u1, u2, cbf_alpha, clf_gamma, penalty_slack_cbf, penalty_slack_clf, h, coeffs_dhdx, v, coeffs_dvdx
        )

        # Solve QP
        prob = osqp.OSQP()
        prob.setup(P, q, A, lb, ub, verbose=False, time_limit=0)
        # Solve QP problem
        res = prob.solve()

        safe_u1, safe_u2, _, _ = res.x
        return np.array([safe_u1, safe_u2])
