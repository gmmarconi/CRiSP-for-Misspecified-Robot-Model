import numpy as np
import math
from numpy.random import default_rng

def generate_circumference(center, radius, num_points):
    """
    Generates a circle trajectory
    :param center: coordinates of circumference center
    :param radius: circumference radius
    :param num_points: number of points in the trajectory
    :return: a ndarray with the trajectory of points
    """
    theta_z = np.linspace(0, 360, num=num_points)
    Circle = np.zeros((num_points, 3))
    Circle[:, 0] = center[0] + radius * np.cos(np.radians(theta_z))
    Circle[:, 1] = center[1] + radius * np.sin(np.radians(theta_z))
    Circle[:, 2] = center[2]

    return Circle

def generate_noisy_circumference_samples(center, radius1, radius2, num_points):
    """
    Generates a circle trajectory
    :param center: coordinates of circumference center
    :param radius1: Main circumference radius
    :param radius2: small torus radius
    :param num_points: number of points to sample
    :return: a ndarray with the trajectory of points
    """

    sample = np.zeros((num_points, 3))

    # Initialize random number generator
    rng = default_rng()

    for i in range(num_points):

        # Uniformly sample joint_ranges
        theta1 = rng.uniform(0, 2*np.pi, 1)
        theta2 = rng.uniform(0, 2*np.pi, 1)
        r = radius2*np.sqrt(rng.uniform(0, 1, 1))

        # Sample on main circumference
        sample[i, 0] = center[0] + radius1 * np.cos(theta1)
        sample[i, 1] = center[1] + radius1 * np.sin(theta1)
        sample[i, 2] = center[2]

        # Add torus noise (see https://stackoverflow.com/a/50746409)
        sample[i, 0] = sample[i, 0] + r * np.cos(theta2) * np.cos(theta1)
        sample[i, 1] = sample[i, 1] + r * np.cos(theta2) * np.sin(theta1)
        sample[i, 2] = sample[i, 2] + r * np.sin(theta2)

    return sample

def generate_spiral(center, radius, theta_max, height, num_points):
    """
    Generates a spiral trajectory
    :param center: coordinates of the center of the containing cylinder
    :param radius: spiral radius
    :param theta_max: maximum angle of the spiral trajectory (0, +inf)
    :param height: height of the spiral along z
    :param num_points: number of points in the trajectory
    :return: a ndarray with the trajectory of points
    """
    theta_z = np.linspace(0, theta_max, num=num_points)

    spiral = np.zeros((num_points, 3))
    spiral[:, 0] = center[0] + radius * np.cos(np.radians(theta_z))
    spiral[:, 1] = center[1] + radius * np.sin(np.radians(theta_z))
    spiral[:, 2] = np.concatenate( (np.linspace(center[2] - height/2.0, center[2] + height/2.0, math.floor(num_points/2)),
                                    np.linspace(center[2] + height/2.0, center[2] - height/2.0, math.ceil(num_points/2))))

    return spiral


def generate_8_shape(center, radius, num_points):
    """
    Generates an eight-like  trajectory
    :param center: coordinates of shape center
    :param radius: shape radius
    :param num_points: number of points in the trajectory
    :return: a ndarray with the trajectory of points
    """
    t = np.linspace(0, 2*np.pi, num_points)
    z = center[2] * np.ones(num_points)
    return np.vstack((radius * np.sin(t) * np.cos(t) + center[0],
                    radius * np.sin(t) + center[1],
                    z
                    )).T


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
        sqx         = np.array(X1*X1, dtype=np.float32)
        rows_X1     = 1
    else:
        sqx = np.sum(np.multiply(X1, X1), 1)
        rows_X1     = sqx.shape[0]

    if X2.ndim <= 1:
        sqy         = np.array(X2*X2, dtype=np.float32)
        rows_X2     = 1
    else:
        sqy = np.sum(np.multiply(X2, X2), 1)
        rows_X2     = sqy.shape[0]
    X1_squares      = np.squeeze(np.outer(np.ones(rows_X1), sqy.T))
    X2_squares      = np.squeeze(np.outer(sqx, np.ones(rows_X2)))
    double_prod     = np.squeeze(2 * np.dot(X1,X2.T))

    return np.maximum(X1_squares + X2_squares - double_prod, np.zeros(X1_squares.shape))

def check_joint_limits_respected(lower_joint_limits, upper_joint_limits, joints_configuration_query):
    """
    Generates a circle trajectory
    :param lower_joint_limits: upper joint limits
    :param upper_joint_limits: lower joint limits
    :param joints_configuration_query: joints configuration to be checked
    :return: True if respected, False otherwise
    """

    # Check list sizes match
    assert len(lower_joint_limits) == len(upper_joint_limits) == len(joints_configuration_query)

    size = len(lower_joint_limits)

    is_ok = True
    for i in range(size):
        is_ok = lower_joint_limits[i] <= joints_configuration_query[i] <= upper_joint_limits[i]
        if not is_ok:
            break

    return is_ok


def compute_rmse(trajectory_1, trajectory_2):

    if trajectory_1.ndim == 2 and trajectory_2.ndim == 2:
        m1 = trajectory_1.shape[0]
        m2 = trajectory_2.shape[0]
        assert m1 == m2
    elif trajectory_1.ndim == 1 and trajectory_2.ndim == 1:
        m1 = 1
        assert trajectory_1.shape[0] == trajectory_2.shape[0]
    else:
        return np.nan


    mse = 0

    for i in range(m1):
        mse += np.linalg.norm(trajectory_1[i] - trajectory_2[i]) ** 2

    mse /= m1

    rmse = np.sqrt(mse)

    # TODO: vectorized form
    # error = np.linalg.norm(trajectory_1 - trajectory_2)
    # mse = np.sqrt(np.mean(np.sum(error ** 2))
    #               )

    return rmse
