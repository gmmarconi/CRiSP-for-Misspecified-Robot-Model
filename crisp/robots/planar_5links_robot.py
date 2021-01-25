from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng
import pickle
from matplotlib import colors
import itertools
from tqdm import tqdm
from crisp.utils.planar_utils import sqdist, radial_squared_error
import pybullet as p
import pybullet_data as pd

class planar_5links_robot(object):
    # Constructor
    def __init__(self,
                 gui=False,
                 urdf_path="./urdf/planar_5R_robot.urdf"):
        """
        Planar 5R robot constructor.

        Parameters: TODO
        """

        # Set up pybullet sim
        self.bullet_client = p
        self.bullet_client.connect(self.bullet_client.DIRECT)
        self.bullet_client.setAdditionalSearchPath(pd.getDataPath())
        self.timeStep = 1. / 60.
        self.bullet_client.setTimeStep(self.timeStep)
        self.bullet_client.setGravity(0, 0, 0)
        num_sim_steps = 5000
        rest_configuration = [0., 1.5708, 0., 4.7124, 0.]
        base_orientation = [-0.707107, 0.0, 0.0, 0.707107]
        # Instantiate simulated Panda object with
        base_offset = [0, 0, 0]  # Base position offset [0,0,0] (meters)
        use_simulation = False  # Do not simulate the physics of the step or force new configuration

        self.base_offset = np.array(base_offset)
        self.base_orientation = base_orientation
        self.use_simulation = use_simulation
        self._gui = gui
        self.num_fixed_joints = 0
        self._urdf_path = urdf_path

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        # Load robot model from URDF
        self.robot_id = self.bullet_client.loadURDF(urdf_path,
                                                    np.array([0, 0, 0]) + self.base_offset,
                                                    self.base_orientation,
                                                    useFixedBase=True,
                                                    flags=flags)

        # Robot-specific parameters
        self._end_effector_index = 5
        self.num_joints = self.bullet_client.getNumJoints(self.robot_id)
        self.link_lengths = [2]*self.num_joints
        self._rest_configuration = [np.pi/2, -np.pi/2, 0, -np.pi/2, 0, 0]

        # Inverse Kinematics configuration parameters
        self.useNullSpace = 1
        self.ikSolver = 0

        # Initialize lists to store joint limits from URDF
        self.lower_joint_limits = []
        self.upper_joint_limits = []

        # Initialize Panda in rest joint configuration
        index = 0
        for j in range(self.num_joints):

            # Get joint limits from URDF
            info = self.bullet_client.getJointInfo(self.robot_id, j)
            jointType = info[2]

            # Set joint damping to zero
            self.bullet_client.changeDynamics(self.robot_id, j, linearDamping=0, angularDamping=0)

            if jointType == self.bullet_client.JOINT_FIXED:
                self.num_fixed_joints += 1

            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.robot_id, j, self._rest_configuration[index])
                self.lower_joint_limits.append(info[8])
                self.upper_joint_limits.append(info[9])
                index = index + 1

        # Compute joint ranges from limits
        self.joint_ranges = []
        zip_object = zip(self.upper_joint_limits, self.lower_joint_limits)
        for ul_i, ll_i in zip_object:
            self.joint_ranges.append(ul_i - ll_i)

        # Initialize joint configuration array
        self.num_non_fixed_joints = self.num_joints - self.num_fixed_joints
        self.joints_configuration = self._rest_configuration

        return

    def getJointsConfiguration(self):
        """
        Returns the current joints configuration.

        Returns:
            joints_configuration: A Python list containing the current joint values
        """

        self.joints_configuration = []
        for j in range(self.num_non_fixed_joints):
            # Get joint limits from URDF
            jointState = self.bullet_client.getJointState(self.robot_id, j)
            self.joints_configuration.append(jointState[0])

        return self.joints_configuration

    def setJointsConfiguration(self, desired_joints_configuration):
        """
        Sets the robot joints configuration to the desired one. Checks for joint limits and prints a warning if violated.

        Parameters:
            desired_joints_configuration: A Python list containing the desired joints configuration.

        Returns:
            attained_joints_configuration: A Python list containing the attained joints configuration.
        """

        # Check dimensions
        assert(len(desired_joints_configuration) == self.num_non_fixed_joints)

        # Set desired joint positions
        for j in range(self.num_non_fixed_joints):
            self.bullet_client.resetJointState(self.robot_id,
                                               j,
                                               desired_joints_configuration[j])

        attained_joints_configuration = desired_joints_configuration

        return attained_joints_configuration

    def get3DEndEffectorPose(self):
        """
        Returns the end-effector pose corresponding to the current joint configuration.

        Parameters:
            None

        Returns:
            ee_pos: a numpy array containing the 3D position of the end effector with respect to the world frame
            ee_orn: a numpy array containing the orientation of the end effector with respect to the world frame, expressed in Euler angles
        """

        # Initialize np arrays to be filled and returned
        ee_pos = np.nan * np.zeros([1, 3])
        ee_orn = np.nan * np.zeros([1, 3])

        # Get position and orientation of the end effector link
        ee_state = self.bullet_client.getLinkState(self.robot_id,
                                                   self._end_effector_index,
                                                   computeForwardKinematics=True)
        ee_pos = np.array(ee_state[4])
        ee_orn = np.array(self.bullet_client.getEulerFromQuaternion(np.array(ee_state[5]).tolist()))

        return ee_pos, ee_orn

    def computeForwardKinematics(self, joint_configurations):
        """
        Returns the end-effector pose corresponding to the current joint configuration.

        Parameters:
            joint_configurations: A numpy matrix with a desired joint configuration for each row.

        Returns:
            A list of numpy matrices containing the corresponding end effector 3D positions and orientations in Euler angles
        """

        joint_configurations = np.atleast_2d(joint_configurations)
        assert(joint_configurations.shape[0] > 0)
        assert(joint_configurations.shape[1] == self.num_non_fixed_joints)

        # Save current joint configuration
        current_config = self.getJointsConfiguration()

        # Initialize np arrays to be filled and returned
        ee_pose = np.nan * np.zeros([joint_configurations.shape[0], 3])

        # Add config for ghost joint
        # joint_configurations = np.hstack((joint_configurations, np.zeros(joint_configurations.shape[0])))

        for idx, q in enumerate(joint_configurations):

            # Set i-th joints configuration
            self.setJointsConfiguration(q)

            # Get position and orientation of the end effector link
            ee_state = self.bullet_client.getLinkState(self.robot_id,
                                                       self._end_effector_index,
                                                       computeForwardKinematics=True)
            ee_pose[idx, :2] = np.array(ee_state[4])[:2]
            ee_pose[idx, 2] = np.array(self.bullet_client.getEulerFromQuaternion(np.array(ee_state[5])))[2]
            #
            # ee_pose[idx, :2] = np.array(ee_state[4])[:2] + [np.sin(q[-1])*self.link_lengths[-1], np.cos(q[-1])*self.link_lengths[-1]]
            # ee_pose[idx, 2] = np.array(self.bullet_client.getEulerFromQuaternion(np.array(ee_state[5]).tolist()))[2]

        # Restore initial joints configuration
        self.setJointsConfiguration(current_config)
        return ee_pose

    def computeInverseKinematics(self, end_effector_poses_2d):
        """
        Computes a joint configuration solution corresponding to a desired end-effector pose. It uses PyBullet's default IK solver (damped least squres).

        Parameters:
            end_effector_poses: A numpy matrix with a desired end-effector pose for each row.

        Returns:
            joint_configurations: A list of numpy matrices containing the corresponding joint configurations computed via PyBullet IK
        """
        joint_configurations = np.nan * np.zeros((end_effector_poses_2d.shape[0],
                                                 self.num_non_fixed_joints))

        end_effector_poses = np.zeros((end_effector_poses_2d.shape[0],
                                        6))

        end_effector_poses[:, :2] = end_effector_poses_2d[:, :2]
        end_effector_poses[:, 3] = -(np.pi/2) * np.ones(end_effector_poses_2d.shape[0])
        end_effector_poses[:, 5] = end_effector_poses_2d[:, 2]

        for idx, ee_pose in tqdm(enumerate(end_effector_poses)):
            ee_target_pos_curr = ee_pose[:3].tolist()
            ee_target_orn_curr = ee_pose[3:].tolist()

            joint_configurations[idx, :] = self.bullet_client.calculateInverseKinematics(
                self.robot_id,
                self._end_effector_index,
                ee_target_pos_curr,
                self.bullet_client.getQuaternionFromEuler(ee_target_orn_curr),
                self.lower_joint_limits,
                self.upper_joint_limits,
                self.joint_ranges,
                self._rest_configuration,
                maxNumIterations=20000,
                residualThreshold=0.00001)

        return joint_configurations

    def reset(self):
        """
        Resets the robot joints configuration to the rest configuration.
        """
        self.setJointsConfiguration(self._rest_configuration)
        return

    def step(self):
        pass

    def get_boundaries(self):
        return [(l, u) for l, u in zip(self.lower_joint_limits, self.upper_joint_limits)]

    def test_trajectory(self,
                        trajectory,
                        true_kinematics_model,
                        model,
                        output_folder=Path('.'),
                        traj_name=None,
                        analyze_preds=False,
                        plot_bias=False,
                        id_string=None,
                        save_svg=False,
                        giffable=False,
                        timestring=None):
        """
        Reconstruct a sequence of orientations in space (a trajectory), one at a time by
        predicting the joint configuration with the supplied model
        :param trajectory: numpy array of orientations
        :param model: a trained model that can has a predict() function
        :param output_folder: folder where to store results
        :param y0: initial joint configuration
        :param orientation: if True, also reconstructs orientations
        :param traj_name: name of the trajectory for visualization purposes
        """
        if giffable:
            (output_folder / 'giffy').mkdir(exist_ok=True,parents=True)
        if analyze_preds:
            (output_folder / f'alpha_plots/{id_string}').mkdir(parents=True, exist_ok=True)
        traj_pts = trajectory.shape[0]
        # Plot circle
        fig = plt.figure( figsize=(9,7))
        # plt.clf()
        f = fig.add_subplot(111)
        # plt.tight_layout()
        f.scatter(trajectory[:, 0], trajectory[:, 1], c='green', s=200, label='Original trajectory')
        x_min = min(np.min(trajectory[:, 0]), 0) - 4
        x_max = max(np.max(trajectory[:, 0]), 0) + 1
        y_min = min(np.min(trajectory[:, 1]), 0) - 1
        y_max = max(np.max(trajectory[:, 1]), 0) + 1
        #
        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        Error = np.zeros((traj_pts, 3), float)
        if plot_bias:
            Error_bias =  np.zeros((traj_pts, 3), float)

        predictions, model_data = model.predict(X=trajectory, is_sequence=True)
        any_outside, preds_outside, out_of_bounds_errors = self.check_if_inside_boundaries(predictions)
        if any_outside:
            print(f"There were {len(preds_outside)} predictions out of range with"
                  f"\taverage error: {np.rad2deg(np.mean(out_of_bounds_errors['average'])):.2f}°"
                  f"\taverage Max error: {np.rad2deg(np.mean(out_of_bounds_errors['worst'])):.2f}°")
        poses = true_kinematics_model.computeForwardKinematics(predictions)
        X_hat = poses[:, 0]
        Y_hat = poses[:, 1]
        theta_hat = poses[:, 2]
        Error[:, :2] = np.abs(trajectory[:, :2] - np.atleast_2d([X_hat, Y_hat]).T)
        Error[:, 2] = np.amin([np.abs(trajectory[:, 2] - theta_hat),
                               2 * np.pi - np.abs(trajectory[:, 2] - theta_hat)])

        joints_pos = true_kinematics_model.get_joints_and_ee_pos_pb(predictions)

        if not giffable:
            f = self.plot_multiple_arms(joints_pos, f, c_links='red', ls='-', alpha=0.2)
            f.scatter(X_hat[0], Y_hat[0], c='red', zorder=1, s=140, edgecolor='k', label='Predicted trajectory')
            f.scatter(X_hat[1:], Y_hat[1:], c='red', zorder=2, s=180, edgecolor='k')
            # plt.show()

        else:
            print("Producing gif frames...")
            for idx, (prediction, test_point, jp) in enumerate(zip(predictions, trajectory, joints_pos)):
                f = self.plot_arm(jp, f, c_links='red', ls='-', alpha=0.2)
                if idx == 0:  # put a label on red markers
                    f.scatter(X_hat, Y_hat, c='red', zorder=1, s=140, edgecolor='k', label='Predicted trajectory')
                else:
                    f.scatter(X_hat, Y_hat, c='red', zorder=2, s=180, edgecolor='k')
                plt.savefig(output_folder / f'giffy/frame_{idx:08d}')
            print("...gif frames completed!")

        plt.legend()
        if id_string:
            plt.savefig(output_folder / f'[{traj_name}] Reconstructed - {id_string}.png')
            if save_svg:
                plt.savefig(output_folder / f'[{traj_name}] Reconstructed - {id_string}.svg')
        else:
            plt.savefig(output_folder / f'[{traj_name}] Reconstructed.png')
            if save_svg:
                plt.savefig(output_folder / f'[{traj_name}] Reconstructed - {id_string}.svg')

        err = np.sqrt(Error ** 2)
        if err.shape[1] == 3:
            mae_orientation = np.mean(err[:, 2])
            var_orientation = np.std(err[:, 2])
        else:
            mae_orientation = None
            var_orientation = None
        var_position = np.std(np.sum(err[:, :2], axis=1))
        mae_position = np.mean(np.sum(err[:, :2], axis=1))
        mae = np.mean(np.sum(err, axis=1))
        var = np.std(np.sum(err, axis=1))

        return {'rmse_orientation': mae_orientation, 'var_orientation': var_orientation,
                'rmse_position': mae_position, 'var_position': var_position,
                'rmse': mae, 'var': var, 'predictions': predictions,
                'pointwise_error': err
                }

    def test_trajectory_inv_pb(self,
                               true_kinematics_model,
                               trajectory, output_folder=Path('.'),
                               traj_name=None,
                               plot_bias=False,
                               id_string=None,
                               giffable=False):
        """
        Reconstruct a sequence of orientations in space (a trajectory), one at a time by
        predicting the joint configuration with the supplied model
        :param trajectory: numpy array of orientations
        :param model: a trained model that can has a predict() function
        :param output_folder: folder where to store results
        :param y0: initial joint configuration
        :param orientation: if True, also reconstructs orientations
        :param traj_name: name of the trajectory for visualization purposes
        """
        if giffable:
            (output_folder / 'giffy_inv_pb').mkdir(exist_ok=True, parents=True)
        traj_pts = trajectory.shape[0]
        # Plot circle
        fig = plt.figure( figsize=(9,7))
        # plt.clf()
        f = fig.add_subplot(111)
        # plt.tight_layout()
        f.scatter(trajectory[:, 0], trajectory[:, 1], c='green', s=200, label='Original trajectory')
        x_min = min(np.min(trajectory[:, 0]), 0) - 1.5
        x_max = max(np.max(trajectory[:, 0]), 0) + 2
        y_min = min(np.min(trajectory[:, 1]), 0) - 2
        y_max = max(np.max(trajectory[:, 1]), 0) + 1.5

        plt.xlim([x_min, x_max])
        plt.ylim([y_min, y_max])

        penalty = 0
        rounds = 0
        q = 0
        misses = 0
        wrong_pts = []

        Error = np.zeros((traj_pts, 3), float)
        Qs_hat = np.zeros((traj_pts, 5), float)
        if plot_bias:
            Error_bias = np.zeros((traj_pts, 3), float)

        predictions = self.computeInverseKinematics(trajectory)

        if not giffable:
            poses = true_kinematics_model.computeForwardKinematics(predictions)
            X_hat = poses[:, 0]
            Y_hat = poses[:, 1]
            theta_hat = poses[:, 2]

            Error[:, :2] = np.abs(trajectory[:, :2] - np.atleast_2d([X_hat, Y_hat]).T)
            Error[:, 2] = np.amin([np.abs(trajectory[:, 2] - theta_hat),
                                     2 * np.pi - np.abs(trajectory[:, 2] - theta_hat)])
            joints_pos = true_kinematics_model.get_joints_and_ee_pos_pb(predictions)
            f = self.plot_multiple_arms(joints_pos, f, c_links='red', ls='-', alpha=0.2)
            f.scatter(X_hat[0], Y_hat[0], c='red', zorder=1, s=140, edgecolor='k', label='Predicted trajectory')
            f.scatter(X_hat[1:], Y_hat[1:], c='red', zorder=2, s=180, edgecolor='k')

        else:
            for idx, (prediction, test_point) in enumerate(zip(predictions, trajectory)):

                if not self.check_if_inside_boundaries(prediction)[0]:
                    misses += 1
                    wrong_pts.append(idx)

                poses = true_kinematics_model.computeForwardKinematics(prediction)
                X_hat = poses[:, 0]
                Y_hat = poses[:, 1]
                theta_hat = poses[:, 2]

                Error[idx, :2] = np.abs(test_point[:2] - np.atleast_2d([X_hat, Y_hat]).T)
                Error[idx, 2] = np.amin([np.abs(test_point[2]-theta_hat),
                                       2*np.pi-np.abs(test_point[2]-theta_hat)])
                Qs_hat[idx] = prediction
                joints_pos = np.squeeze(true_kinematics_model.get_joints_and_ee_pos_pb(prediction))
                f = self.plot_arm(joints_pos, f, c_links='red', ls='-', alpha=0.2)

                if idx == 0:  # put a label on red markers
                    f.scatter(X_hat, Y_hat, c='red', zorder=1, s=140, edgecolor='k', label='Predicted trajectory')
                else:
                    f.scatter(X_hat, Y_hat, c='red', zorder=2, s=180, edgecolor='k')
                plt.savefig(output_folder / f'giffy_inv_pb/frame_{idx:08d}')

        # if fig_text is not None:
        #     plt.plot([], [], ' ', label=fig_text)
        plt.legend()
        if id_string:
            plt.savefig(output_folder / f'[{traj_name}] Pybullet inverse kinematics - {id_string}.png')
        else:
            plt.savefig(output_folder / f'[{traj_name}] Pybullet inverse kinematics.png')


        err = np.sqrt(Error ** 2)
        if err.shape[1] == 3:
            mae_orientation = np.mean(err[:, 2])
            var_orientation = np.std(err[:, 2])
        else:
            mae_orientation = None
            var_orientation = None
        var_position = np.std(np.sum(err[:, :2], axis=1))
        mae_position = np.mean(np.sum(err[:, :2], axis=1))
        mae = np.mean(np.sum(err, axis=1))
        var = np.std(np.sum(err, axis=1))

        bias_errors = {}
        if plot_bias:
            Error_bias = np.sqrt(Error_bias**2)
            bias_errors['rmse_orientation'] = np.mean(Error_bias[:, 2])
            bias_errors['var_orientation'] = np.std(Error_bias[:, 2])
            bias_errors['rmse_position'] = np.mean(np.sum(Error_bias[:, :2], axis=1))
            bias_errors['var_position'] = np.std(np.sum(Error_bias[:, :2], axis=1))
            bias_errors['rmse'] = np.mean(np.sum(Error_bias, axis=1))
            bias_errors['var'] = np.std(np.sum(Error_bias, axis=1))

        return {'rmse_orientation': mae_orientation, 'var_orientation': var_orientation,
                'rmse_position': mae_position, 'var_position': var_position,
                'rmse': mae, 'var': var, 'predictions': predictions,
                'pointwise_error': err}

    def clip_into_boundaries(self, Q):
        """ Clips inside joint boundaries the supplied numpy array"""
        for idx, (q, low, high) in enumerate(zip(Q, self.lower_joint_limits, self.upper_joint_limits)):
            if q < low:
                Q[idx] = low
            elif q > high:
                Q[idx] = high
        # return Q

    def check_if_inside_boundaries(self, Qs):
        """ Checks if the supplied numpy array is a valid joint configuration"""
        outside = []
        errors = {"average":[], "worst": []}
        lower_joint_limits = np.array(self.lower_joint_limits)
        upper_joint_limits = np.array(self.upper_joint_limits)
        Qs = np.atleast_2d(Qs)

        for idx, Q in enumerate(Qs):
            below_idx = Q < lower_joint_limits
            above_idx = Q > upper_joint_limits

            if below_idx.any() or above_idx.any():
                mean_errs = []
                max_errs = []
                if below_idx.any():
                    mean_errs.append(np.mean(lower_joint_limits[below_idx] - Q[below_idx]))
                    max_errs.append(np.max(lower_joint_limits[below_idx] - Q[below_idx]))
                if above_idx.any():
                    mean_errs.append(np.mean(Q[above_idx] - lower_joint_limits[above_idx]))
                    max_errs.append(np.max(Q[above_idx] - lower_joint_limits[above_idx]))
                errors['average'].append(np.mean(mean_errs))
                errors['worst'].append(np.max(max_errs))
                outside.append(idx)

                if below_idx.any():
                    Q[below_idx] = lower_joint_limits[Q < lower_joint_limits]
                if above_idx.any():
                    Q[above_idx] = upper_joint_limits[Q > upper_joint_limits]

        if outside:
            return True, outside, errors
        else:
            return False, outside, errors

    def generate_dset(self, samples, save_folder=Path('.'), savename=None):
        """
         Generates a dataset of points randomly sampled for the joint space of the robot
         :param samples: number of samples to generate
         :param output_folder: folder where to save .pickle file
         :param remove_duplicates: if True, removes all duplicated points
         :return: a dict with keys 'xtr', 'xval', 'xte', 'ytr', 'yval, 'yte' for the generated dataset
         """
        save_folder.mkdir(exist_ok=True, parents=True)
        rng = default_rng()
        Qs = rng.random([samples, self.num_non_fixed_joints])
        high_tiled = np.tile(np.array(self.upper_joint_limits), (samples, 1))
        low_tiled = np.tile(np.array(self.lower_joint_limits), (samples, 1))
        Qs = (high_tiled - low_tiled) * Qs + low_tiled  # Generates samples in the desired intervals

        poses = self.computeForwardKinematics(Qs)
        X = poses[:, 0]
        Y = poses[:, 1]
        Theta = poses[:, 2]

        plt.plot([X, X + 0.2 * np.sin(Theta)],
                 [Y, Y - 0.2 * np.cos(Theta)], 'k-')
        plt.scatter(X, Y)  # Plotting the data set
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.savefig(save_folder / f'Data set of {samples} possible endeffector positions and orientations.png')
        output = Qs  # Q1,Q2,Q3

        return self.split_data_into_train_val_test(poses, output, save_folder, savename=savename)

    def generate_dset_near_trajectory(self, trajectory, max_dist, samples,
                                      orientation=False,
                                      savename=None,
                                      save_folder=Path('.')):
        """
        Samples training points close to the supplied trajectory, if remove_duplicates is True,
        it removes duplicated points
        :return: dict with dataset
        """
        Qs = []
        rng = default_rng()
        tentative_per_try = 50000
        tries = 0
        if orientation:
            check_every = tentative_per_try*20
        else:
            check_every = tentative_per_try*5
        while len(Qs) <= samples:
            tries += tentative_per_try
            tentative_Qs = rng.random([tentative_per_try, self.num_non_fixed_joints])
            high_tiled = np.tile(np.array(self.upper_joint_limits), (tentative_per_try, 1))
            low_tiled = np.tile(np.array(self.lower_joint_limits), (tentative_per_try, 1))
            tentative_Qs = (high_tiled - low_tiled) * tentative_Qs + low_tiled

            poses = self.computeForwardKinematics(tentative_Qs)
            X = poses[:, 0]
            Y = poses[:, 1]
            distances = sqdist(X1=np.vstack((X, Y)).T, X2=trajectory[:, :2])

            if orientation:
                Theta = poses[:, 2]
                circle_distances_min = np.array([np.sqrt(radial_squared_error(np.atleast_2d(candidate).T, np.atleast_2d(trajectory[:, 2]))).min() for candidate in Theta])
                [Qs.append(q) for (q, p_wise_dists, min_ori_dist) in zip(tentative_Qs, distances, circle_distances_min)
                 if (p_wise_dists.min() <= max_dist) and (min_ori_dist <= 0.03)]

            else:
                [Qs.append(q) for (q, p_wise_dists) in zip(tentative_Qs, distances) if p_wise_dists.min() <= max_dist]

            if tries % check_every == 0:
                print(f'Generating dataset.\t Tries: {tries}\tCurrent samples: {len(Qs)}\\{samples}')

        if len(Qs) > samples:
            [Qs.pop() for idx in range(len(Qs)-samples)]

        Qs = np.array(Qs)
        poses = self.computeForwardKinematics(Qs)
        X = poses[:, 0]
        Y = poses[:, 1]
        Theta = poses[:, 2]

        # plt.plot([X, X + 0.2 * np.cos(Theta)],
        #         [Y, Y + 0.2 * np.sin(Theta)], 'k-')
        # plt.scatter(X, Y)  # Plotting the data set
        # plt.xlabel("X Axis")
        # plt.ylabel("Y Axis")
        # plt.savefig(output_folder / f'Dataset of {samples} samples with {traj_name} trajectory.png')

        data = np.vstack((X, Y, Theta)).T  # X,Y,Theta
        output = np.array(Qs)

        return self.split_data_into_train_val_test(data, output, save_folder, fullspace=False, savename=savename)

    @classmethod
    def split_data_into_train_val_test(cls, inputs, outputs, save_folder=Path('.'),
                                       savename=None, fullspace=True):
        """ Separate data set in to Train, Test And Validation """
        samples = inputs.shape[0]
        train_input = inputs[0:int(0.7 * samples), :]
        train_output = outputs[0:int(0.7 * samples), :]

        test_input = inputs[int(0.7 * samples):int(0.85 * samples), :]
        test_output = outputs[int(0.7 * samples):int(0.85 * samples), :]

        validate_input = inputs[int(0.85 * samples):int(samples), :]
        validate_output = outputs[int(0.85 * samples):int(samples), :]

        dataset = {'xtr': train_input, 'ytr': train_output,
                   'xte': test_input, 'yte': test_output,
                   'xval': validate_input, 'yval': validate_output}

        save_folder.mkdir(exist_ok=True, parents=True)
        if savename is not None:
            pickle.dump(dataset, open(save_folder / savename, "wb"))
        elif fullspace:
            pickle.dump(dataset, open(save_folder / f'synth_dset_{samples}_fullspace.pickle', 'wb'))
        else:
            pickle.dump(dataset, open(save_folder / f'synth_dset_{samples}_neartraj.pickle', 'wb'))

        return dataset


    def get_joints_and_ee_pos_pb(self, joint_configurations):
        joint_configurations = np.atleast_2d(joint_configurations)
        num_configurations = joint_configurations.shape[0]
        positions = self.num_joints # We add the position for the end effector
        assert(joint_configurations.shape[1] == self.num_non_fixed_joints)
        joints_pos = np.full(((num_configurations, positions, 2)), np.inf)

        for idx, jc in enumerate(joint_configurations):
            # Set i-th joints configuration
            self.setJointsConfiguration(jc)

            # Get position and orientation of the end effector link
            for link_idx in range(self.num_non_fixed_joints):
                ee_state = self.bullet_client.getLinkState(self.robot_id,
                                                       link_idx,
                                                       computeForwardKinematics=True)
                joints_pos[idx, link_idx] = np.array(ee_state[4])[:2]

            P = self.computeForwardKinematics(np.atleast_2d(jc)).squeeze()
            joints_pos[idx, -1] = P[:2]
            # Restore initial joints configuration
        return joints_pos

    @staticmethod
    def plot_arm(extr, f, c_links='red', c_joints='black', **kwargs):
        """Plots the arms given the link extremities positions in a [N_l x 2] array"""
        plot_args = {key: value for (key, value) in kwargs.items() if key != 'label'}
        for idx, _ in enumerate(extr):
            # first plot is to assign only one label using **kwargs
            if idx == 0:
                # plot links
                f.plot([0, extr[idx, 0]], [0, extr[idx, 1]], c=c_links, **kwargs)
                # plot joints
                f.scatter(extr[idx, 0], extr[idx, 1],  marker='s', c=c_joints, alpha=1)
            else:
                #plot links
                f.plot([extr[idx-1, 0], extr[idx, 0]],
                       [extr[idx-1, 1], extr[idx, 1]], '-', c=c_links, **plot_args)
                # plot joints
                f.scatter(extr[idx, 0], extr[idx, 1], marker='o', c=c_joints, alpha=1, zorder=2)
        return f

    @staticmethod
    def plot_multiple_arms(extr, f, c_links='red', c_joints='black', **kwargs):
        """Plots the arms given the link extremities positions in a [N_l x 2 x N_p] array"""
        plot_args = {key: value for (key, value) in kwargs.items() if key != 'label'}
        for idx in range(extr.shape[1]):
            # first plot is to assign only one label using **kwargs
            if idx == 0:
                # plot links
                f.plot([np.zeros(extr.shape[0]), extr[:, idx, 0]], [np.zeros(extr.shape[0]), extr[:, idx, 1]],
                       c=c_links, **kwargs)  # plot joints
                f.scatter(extr[:, idx, 0], extr[:, idx, 1],  marker='s', c=c_joints, alpha=0.3)
            else:
                #plot links
                f.plot([extr[:, idx-1, 0], extr[:, idx, 0]],
                       [extr[:, idx-1, 1], extr[:, idx, 1]], '-', c=c_links, **plot_args)
                # plot joints
                f.scatter(extr[:, idx, 0], extr[:, idx, 1], marker='o', c=c_joints, alpha=0.3, zorder=2)
        return f

    @staticmethod
    def plot_arm_old(extr, f, **kwargs):
        """Plots the arms given the link extremities positions in a [N_l x 2] array"""
        plot_args = {key: value for (key, value) in kwargs.items() if key != 'label'}
        for idx, _ in enumerate(extr):
            if idx == 0:
                f.plot([0, extr[idx, 0]], [0, extr[idx, 1]], zorder=1, **kwargs)
                f.scatter(extr[idx, 0], extr[idx, 1], zorder=2,  marker='o', color='k')
            else:
                f.plot([extr[:idx, 0].sum(), extr[:idx+1, 0].sum()],
                       [extr[:idx, 1].sum(), extr[:idx+1, 1].sum()], '-k', zorder=1, **plot_args)
                f.scatter(extr[:idx, 0].sum(), extr[:idx+1, 0].sum(), zorder=2,  marker='o', color='k')
        return f

    def get_joints_pos_old(self, Q, idx=0):
        if Q.ndim == 1:
            Q = np.atleast_2d(Q)
        joints_pos = np.full((self.num_non_fixed_joints, 2), np.inf)
        for jdx, length in enumerate(self.link_lengths[:self.num_non_fixed_joints]):
            x = length * np.cos(np.sum(np.atleast_2d(Q[idx, :jdx + 1]), axis=1))
            y = length * np.sin(np.sum(np.atleast_2d(Q[idx, :jdx + 1]), axis=1))
            joints_pos[jdx] = (x, y)
        return joints_pos

    def analyze_point(self, xtr, x_true, x_hat, alphas, joints_pos,
                      a_indexes=None,
                      **kwargs):
        """Plots the alphas associated to a certain point"""
        assert x_true.shape[0] < 3, "analyze_point() supports only 2d points"
        assert x_hat.shape[0] < 3, "analyze_point() supports only 2d points"
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlim([-10, -0])
        ax.set_ylim([-8, 8])
        ax.set_title(f'Alphas')

        if a_indexes is not None:
            xtr_to_plot = xtr[a_indexes]
            alphas_to_plot = alphas[a_indexes]
        else:
            xtr_to_plot = xtr
            alphas_to_plot = alphas
        divnorm = colors.DivergingNorm(vmin=alphas.min(), vmax=alphas.max(), vcenter=0)
        splot = ax.scatter(xtr_to_plot[:, 0], xtr_to_plot[:, 1], cmap='PiYG',
                            norm=divnorm, c=alphas_to_plot,
                           alpha=0.4, s=20, zorder=0)
        # splot = ax.scatter(xtr_to_plot[alphas_to_plot>=0, 0], xtr_to_plot[alphas_to_plot>=0, 1], cmap='PiYG',
        #                    vmin=alphas.min(), vmax=alphas.max(), c=alphas_to_plot[alphas_to_plot>=0],
        #                    alpha=0.4, s=30, zorder=1)
        # ax.plot([xtr[:, 0], xtr[:, 0] + 0.2 * np.cos(xtr[:, 2])],
        #         [xtr[:, 1], xtr[:, 1] + 0.2 * np.sin(xtr[:, 2])],
        #         'c-', alpha=0.2, cmap='twilight', colors=xtr[:, 2])
        # cbar = fig.colorbar(splot)
        # cbar.ax.tick_params(labelsize=20)
        ax = self.plot_arm(joints_pos, ax)
        ax.scatter(x_true[0], x_true[1], zorder=2, marker='X', color='blue', s=100, label='True')
        ax.scatter(x_hat[0], x_hat[1], zorder=2, marker='X', color='red', s=100, label='Reconstructed')
        marker = itertools.cycle(['.', 'o', 'v', '^', '1', '2', '3', '4', '8', 's', 'p', 'P', 'h'])
        for key, value in kwargs.items():
            ax.scatter(value[0], value[1], marker=next(marker), zorder=2, s=160, label=key)

        ax.legend(fontsize=12)
        # plt.show()

        return fig, ax

    def plot_configs_of_biggest_alphas(self, xtr, ytr, x_true, x_hat, y_hat, alphas, num_to_plot=3):
        assert x_true.shape[0] < 3, "analyze_point() supports only 2d points"
        assert x_hat.shape[0] < 3, "analyze_point() supports only 2d points"
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_axes([0,0,1,1])
        ax.set_xlim([-6.5, 0.5])
        ax.set_ylim([-2.5, 6])
        ax.set_title(f'Alphas')
        a_indexes = np.argsort(alphas)[-num_to_plot:]
        xtr_to_plot = xtr[a_indexes]
        ytr_to_plot = ytr[a_indexes]

        splot = ax.scatter(xtr_to_plot[:, 0], xtr_to_plot[:, 1], c='tab:orange',
                           s=100, zorder=2)
        vivid_c = 'tab:red'
        dull_c = 'black'
        for y in ytr_to_plot:
            j_pos = self.get_joints_and_ee_pos_pb(y)
            fig = self.plot_arm(j_pos, ax, c_links='black', c_joints=vivid_c, ls='-', alpha=0.7, linewidth=4, zorder=1)

        ax.scatter(x_true[0], x_true[1], zorder=1, marker='X', color='blue', s=1000, label='Target')
        fig = self.plot_arm(self.get_joints_and_ee_pos_pb(y_hat), ax, c_links=vivid_c, c_joints=dull_c, ls='-', alpha=1, linewidth=8, zorder=1)
        ax.scatter(x_hat[0], x_hat[1], zorder=2, marker='X', color='tab:red', s=400, label='Reconstructed')
        ax.legend(fontsize=30)

        return fig, ax

