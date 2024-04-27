import numpy as np
from matplotlib import pyplot as plt


def visualize_quadrotor_simulation_result(
    quadrotor, states: np.ndarray, obs_center: np.ndarray, obs_radius: np.ndarray
):
    def plot_obstacles(obs_center, obs_radius):
        for obs_p, obs_r in zip(obs_center, obs_radius):
            circle = plt.Circle(
                tuple(obs_p),
                obs_r,
                color="grey",
                fill=True,
                linestyle="--",
                linewidth=2,
                alpha=0.5,
            )
            plt.gca().add_artist(circle)

    plt.figure()
    plt.gca().set_aspect("equal", adjustable="box")
    ys = [s[0] for s in states]
    zs = [s[2] for s in states]
    phis = [s[4] for s in states]
    # Generate circle for CBF
    plot_obstacles(obs_center, obs_radius)

    # Plot the trajectory
    plt.scatter(ys, zs)

    # Plot quadrotor pose
    y_ = [(y - quadrotor.l_q * np.cos(phi), y + quadrotor.l_q * np.cos(phi)) for (y, phi) in zip(ys[::10], phis[::10])]
    z_ = [(z - quadrotor.l_q * np.sin(phi), z + quadrotor.l_q * np.sin(phi)) for (z, phi) in zip(zs[::10], phis[::10])]
    for yy, zz in zip(y_, z_):
        plt.plot(yy, zz, marker="o", color="r", alpha=0.5)

    # plot start and end point
    init_x, init_y = states[0][0], states[0][2]
    plt.scatter(init_x, init_y, s=200, color="green", alpha=0.75, label="init. position")
    plt.scatter(5.0, 5.0, s=200, color="purple", alpha=0.75, label="target position")

    plt.xlim(-0.5, 7)
    plt.ylim(-0.5, 7)
    plt.grid()
    plt.show()
