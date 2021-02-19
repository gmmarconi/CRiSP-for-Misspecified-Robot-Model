import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def generate_circle(cx, cy, num_points, r=None):
    """
    Generates a circle trajectory
    :param cx: x coordinate of center
    :param cy: y coordinate of center
    :param r: radius
    :param num_points: number of points in the trajectory
    :return: a ndarray with the trajectory of points
    """
    if r is None:
        r = 1
    theta_z = np.linspace(0, 2*np.pi, num=num_points)
    Circle = np.zeros((num_points, 3))
    Circle[:, 0] = cx + r * np.cos(theta_z)
    Circle[:, 1] = cy + r * np.sin(theta_z)
    Circle[:, 2] = np.pi/8 * np.sin(theta_z) - np.pi/2 #+ 0.1
    # Circle[:, 2] = np.hstack([np.linspace(0, 60, num_points // 4),
    #                           np.linspace(60, 5, num_points // 4),
    #                           np.linspace(5, -30, num_points // 4),
    #                           np.linspace(-30, 0, num_points // 4)])

    return Circle


def generate_8(cx, cy, num_points, scale=None, rotation=None):
    """
    Generates an eight-like  trajectory
    :param cx: x coordinate of center
    :param cy: y coordinate of center
    :param r: radius
    :param num_points: number of points in the trajectory
    :return: a ndarray with the trajectory of points
    """
    if scale is None:
        scale = 1
    t = np.linspace(0, 2*np.pi, num_points)
    trajectory = np.vstack((scale * np.sin(t)*np.cos(t) + cx,
                    scale *np.sin(t) + cy,
                    # 0.20 * np.arctan2(np.sin(t)*np.cos(t) + cx, scale * np.sin(t) + cy)
                    0.2 * np.ones(num_points)
                      )).T
    if rotation:
        # rotations in anticlockwise sense
        rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                               [np.sin(rotation), np.cos(rotation)]])
        trajectory[:, :2] = (rot_matrix @ trajectory[:, :2].T).T
        # trajectory[:, 2] += rotation

    return trajectory


def predict_panda_trajectory(xte, model, forward, out, s, v, alg):
    predictions, data = model.predict(xte, is_sequence=False)

    # Compare results on test set
    reconstructed = forward(predictions)
    error = np.full((xte.shape), np.inf)

    error[:,:3] = np.sqrt((reconstructed[:, :3] - xte[:, :3])**2)
    error[:, 3:] = np.sqrt(radial_squared_error(reconstructed[:, 3:], xte[:, 3:]))
    rmse_orientation = np.mean(np.sum(error[:, 3:], axis=1))
    var_orientation = np.std(np.sum(error[:, 3:], axis=1))
    var_position = np.std(np.sum(error[:, :3], axis=1))
    rmse_position = np.mean(np.sum(error[:, :3], axis=1))
    rmse = np.mean(np.sum(error, axis=1))
    var = np.std(np.sum(error, axis=1))
    results = {'rmse_orientation': rmse_orientation, 'var_orientation': var_orientation,
                'rmse_position': rmse_position, 'var_position': var_position,
                'rmse': rmse, 'var': var, 'predictions': predictions}

    out(f"s: {s}\t v: {v}")
    out(f"\t\tRMSE ori: {results['rmse_orientation']:7.6f} ± {results['var_orientation']:7.6f}")
    out(f"\t\tRMSE pos: {results['rmse_position']:7.6f} ± {results['var_position']:7.6f}\n"
             f"\t\tRMSE: {results['rmse']:7.6f} ± {results['var']:7.6f}")
    out(f"Norm of alphas = {np.linalg.norm(data['alpha'])}")

    out(f"Algorithm: {alg}")

def print_avg_distances_on_dataset(X1, X2, output_folder, dimensions=None,
                                   plot_hist=False,
                                   savename=None,
                                   out=None):
    """
    Plots statistics on daa distances
    :param X1: First matrix of points
    :param X2: Second matrix of points
    :param output_folder: output folder where to save data
    :param dimensions: spatial dimensionality of data: 2D or 3D (excluding orientation)
    :param savename: savename of the pictures
    """

    if savename is None:
        savename = 'hists'
    if out is None:
        out = print

    (output_folder / "hists").mkdir(parents=True, exist_ok=True)

    def print_stats(D):
        out(f"Avg distance 5th nearest neighbour: {np.mean(D_sorted[:, 5]):.5f}  ± {np.std(D_sorted[:, 5]):.5f}\n"
              f"Avg distance 10th nearest neighbour: {np.mean(D_sorted[:, 10]):.5f}  ± {np.std(D_sorted[:, 10]):.5f}\n"
              f"Avg distance 20th nearest neighbour: {np.mean(D_sorted[:, 20]):.5f}  ± {np.std(D_sorted[:, 20]):.5f}\n"
              f"Mean distance: {np.mean(D[D > 0]):.5f} ± {np.std(D[D > 0]):.5f}\n"
              f"Median distance: {np.median(D[D > 0]):.5f}\n"
              f"Avg Min distance: {np.mean(D_sorted[:, 1]):.5f} ± {np.std(D_sorted[:, 1]):.5f}\n"
              f"Avg Max distance: {np.mean(D_sorted[:, -1]):.5f} ± {np.std(D_sorted[:, -1]):.5f}\n")

    if X2 is None:
        X2 = X1

    DD = {}
    D = np.sqrt(sqdist(X1, X2))
    D_sorted = np.sort(D, axis=1)
    out(f"## Overall Distances ##")
    print_stats(D_sorted)
    DD['Overall'] = D

    if dimensions is not None:
        D_pos = np.sqrt(sqdist(X1[:, :dimensions], X2[:, :dimensions]))
        D_sorted = np.sort(D_pos, axis=1)
        out(f"## Position distances ##")
        print_stats(D_sorted)
        DD['Position'] = D_pos

        D_ori = np.sqrt(sqdist(X1[:, dimensions:], X2[:, dimensions:]))
        D_sorted = np.sort(D_ori, axis=1)
        out(f"## Orientation distances ##")
        print_stats(D_sorted)
        DD['Orientation'] = D_ori

    if plot_hist:
        hists, axes = plt.subplots(3, 1)
        # hists.xticks(fontsize=10)
        # hists.yticks(fontsize=10)
        for d, (name, dists) in enumerate(DD.items()):
            axes[d].hist(dists.flatten(), color='skyblue')
            axes[d].set_title(f"{name}")
            axes[d].tick_params(axis='both', labelsize=8)
        plt.savefig(output_folder / 'hists' / f"{savename}")
        plt.close(hists)



def sqdist(X1, X2):
    """ Given two matrices whose rows are points, computes the distances
    between all the points of the first matrix and all the points of the
    second matrix
    Arguments:
    X1: [N1 x d], earch row is a d-dimensional point
    X2: [N2 x d], each row is a d-dimensional point
    Returns:
    M: [N1 x N2], each element  is the distance between two points
    M_ij = || X1_i - X2_j ||"""

    if not isinstance(X1, np.ndarray):
        X1 = np.array(X1)
    if not isinstance(X2, np.ndarray):
        X2 = np.array(X2)
    if X1.ndim <= 1:
        sqx = np.array(X1*X1, dtype=np.float32)
        rows_X1 = 1
    else:
        sqx = np.sum(np.multiply(X1, X1), 1)
        rows_X1 = sqx.shape[0]

    if X2.ndim <= 1:
        sqy = np.array(X2*X2, dtype=np.float32)
        rows_X2 = 1
    else:
        sqy = np.sum(np.multiply(X2, X2), 1)
        rows_X2 = sqy.shape[0]
    X1_squares = np.squeeze(np.outer(np.ones(rows_X1), sqy.T))
    X2_squares = np.squeeze(np.outer(sqx, np.ones(rows_X2)))
    double_prod = np.squeeze(2 * np.dot(X1,X2.T))

    return np.maximum(X1_squares + X2_squares - double_prod, np.zeros(X1_squares.shape))

def radial_squared_error(Y1, Y2):
    """Computes radial distance dimension-wise between two matrices of radial points [N1 x d] and [N2 x d],
    where N1 and N2 are the number of points"""
    assert isinstance(Y1, np.ndarray), "radial_squared_error(): please provide a ndarray"
    assert isinstance(Y2, np.ndarray), "radial_squared_error(): please provide a ndarray"

    return np.amin([np.abs(Y1-Y2), 2*np.pi-np.abs(Y1-Y2)], axis=0)**2

def nneigh(X0, X1, k, verbose=False):
    # Compute the average nearest neighbour distances
    D = np.sqrt(sqdist(X0, X1))
    sorteD = np.sort(D, axis=0)
    kth_n_dist = np.mean(sorteD[:k], axis=1)
    if verbose:
        print(f"Average distance from {k}th nearest neighbour: {kth_n_dist}")
    return kth_n_dist, D

def gausskernel(X1, X2, sigma=1):
    """ Computes the kernel matrix given two matrices whose rows
    are points"""
    return np.exp(-sqdist(X1, X2) / (2*sigma**2))

def sqdist_weighted(X1, X2, S):
    """ Given two matrices whose rows are points, computes the distances
    between all the points of the first matrix and all the points of the
    second matrix
    Arguments:
    X1: [N1 x d], earch row is a d-dimensional point
    X2: [N2 x d], each row is a d-dimensional point
    S: [1 x d] vector with sigmas for each dimension
    Returns:
    M: [N1 x N2], each element  is the distance between two points
    M_ij = || X1_i - X2_j ||"""

    if not isinstance(X1, np.ndarray) or not isinstance(X1, np.ndarray) or X1.ndim <= 1 or X2.ndim <= 1:
        raise ValueError("sqdist_wighted() works only with multidimensional points")

    N1 = X1.shape[0]
    sqx = np.sum(X1**2 / np.tile(S**2, (N1, 1)), 1)

    N2 = X2.shape[0]
    sqy = np.sum(X2**2 / np.tile(S**2, (N2, 1)), 1)

    X1_squares = np.squeeze(np.outer(np.ones(N1), sqy.T))
    X2_squares = np.squeeze(np.outer(sqx, np.ones(N2)))
    weighted_X1 = X1 / np.tile(S ** 2, (N1, 1))
    double_prod = np.squeeze(2 * np.dot(weighted_X1,X2.T))

    return np.maximum(X1_squares + X2_squares - double_prod, np.zeros(X1_squares.shape))


def plot_dataset(xtr, ytr, traj, planar_manip, output_folder):
    fig_dset = plt.figure()
    ax = fig_dset.add_axes([0, 0, 1, 1])
    plt.xlim([-11, 12])
    plt.ylim([-11, 12])

    sel = 1
    config_to_visualize = ytr[sel]
    config_to_visualize = np.array([np.pi, np.pi, np.pi, np.pi, np.pi]) / 4
    ax.scatter(xtr[:, 0], xtr[:, 1], zorder=0, alpha=0.4)
    # # plot orientation of xtr
    ax.plot([xtr[:, 0], xtr[:, 0] + 0.2 * np.sin(xtr[:, 2])],
            [xtr[:, 1], xtr[:, 1] - 0.2 * np.cos(xtr[:, 2])],
            'c-', alpha=0.2, zorder=0)

    # plot trajectory and orientation
    ax.scatter(traj[:, 0], traj[:, 1], c='orange', s=100, zorder=1, alpha=0.8, edgecolor='k')
    ax.plot([traj[:, 0], traj[:, 0] + 0.4 * np.sin(traj[:, 2])],
            [traj[:, 1], traj[:, 1] - 0.4 * np.cos(traj[:, 2])],
            'k-', alpha=1, zorder=2)

    P_hat = planar_manip.computeForwardKinematics(np.atleast_2d(config_to_visualize)).squeeze()
    planar_manip.plot_arm(np.squeeze(planar_manip.get_joints_and_ee_pos_pb(config_to_visualize)),
                          ax, ls='-', linewidth=12, label='manipulator', alpha=0.2, c_links='red')

    X_hat, Y_hat = P_hat[0], P_hat[1]
    ax.scatter(X_hat, Y_hat, c='tab:red', zorder=2, s=600, edgecolor='k',
               label='EE position', marker='X')
    plt.xlabel('X')
    plt.ylabel('Y')
    # ax.set_yticklabels([])
    # ax.set_xticklabels([])
    plt.legend()
    plt.savefig(output_folder / f'dataset.png', format='png')
    plt.show()
    plt.close(fig_dset)


