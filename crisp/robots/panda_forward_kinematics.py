import time
from tqdm import tqdm
import numpy as np
import crisp.utils.panda_utils as iku3d
import warnings

# Panda-specific parameters
panda_end_effector_index = 11  # 8
panda_num_dofs = 7
idx_true_joints = [0, 1, 2, 3, 4, 5, 6, 9, 10]


class PandaSimForwardKin(object):

    # Constructor
    def __init__(self,
                 bullet_client,
                 base_offset,
                 base_orientation,
                 gui=False,
                 use_simulation=True,
                 urdf_path="franka_panda/panda.urdf"):

        self.bullet_client = bullet_client
        self.base_offset = np.array(base_offset)
        self.base_orientation = base_orientation
        self.use_simulation = use_simulation
        self._gui = gui
        self.num_fixed_joints = 0
        self._urdf_path = urdf_path
        self.num_controlled_joints = 0

        # Panda rest pose used for initialization and by null-space IK solver
        #   Note: last 2 scalars are prismatic finger joints and can be ignored
        self.rest_configuration = [0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.02, 0.02]

        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

        # Load Panda arm model from URDF
        self.panda_id = self.bullet_client.loadURDF(urdf_path,
                                                    np.array([0, 0, 0]) + self.base_offset,
                                                    self.base_orientation,
                                                    useFixedBase=True,
                                                    flags=flags)

        # Official Panda joint limits
        # lower limits for null space
        self.lower_joint_limits = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0]
        # upper limits for null space
        self.upper_joint_limits = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04]

        # Initialize lists to store joint limits from URDF
        # self.lower_joint_limits = []
        # self.upper_joint_limits = []

        # Initialize Panda in rest joint configuration
        index = 0
        fingers = 0
        for j in range(self.bullet_client.getNumJoints(self.panda_id)):
            # Set joint damping to zero
            self.bullet_client.changeDynamics(self.panda_id, j, linearDamping=0, angularDamping=0)

            info = self.bullet_client.getJointInfo(self.panda_id, j)
            jointType = info[2]
            if jointType == self.bullet_client.JOINT_FIXED:
                self.num_fixed_joints += 1

            if jointType == self.bullet_client.JOINT_PRISMATIC:
                self.bullet_client.resetJointState(self.panda_id, j, self.rest_configuration[index])
                index += 1
                fingers += 1

            if jointType == self.bullet_client.JOINT_REVOLUTE:
                self.bullet_client.resetJointState(self.panda_id, j, self.rest_configuration[index])
                index += 1

        # Joint ranges for null space IK
        self.joint_ranges = []
        zip_object = zip(self.upper_joint_limits, self.lower_joint_limits)
        for ul_i, ll_i in zip_object:
            self.joint_ranges.append(ul_i - ll_i)

        # Initialize simulation time counter to zero
        self.t = 0.

        self.num_non_fixed_joints = self.bullet_client.getNumJoints(self.panda_id) - self.num_fixed_joints
        self.num_controlled_joints = self.num_non_fixed_joints - fingers

    def reset(self):
        pass

    def compute_forward_kin(self, joint_configurations):
        # IN: A numpy matrix with a desired joint configuration for each row.
        # OUT: A list of numpy matrices containing the corresponding end effector 3D positions and orientations in Euler angles

        joint_configurations = np.atleast_2d(joint_configurations)
        assert(joint_configurations.shape[0] > 0)
        assert(joint_configurations.shape[1] == self.num_controlled_joints)

        current_config = self.getJointsConfiguration()

        # Initialize np arrays to be filled and returned
        ee_pos = np.nan * np.zeros([joint_configurations.shape[0], 3])
        ee_orn = np.nan * np.zeros([joint_configurations.shape[0], 3])

        joint_configurations = np.hstack((joint_configurations, np.zeros((joint_configurations.shape[0], 2))))

        for idx, q in enumerate(joint_configurations):
            self.setJointsConfiguration(q)

            ee_state = self.bullet_client.getLinkState(self.panda_id,
                                                       panda_end_effector_index,
                                                       computeForwardKinematics=True)
            ee_pos[idx] = np.array(ee_state[4])
            ee_orn[idx] = np.array(self.bullet_client.getEulerFromQuaternion(np.array(ee_state[5])))
            #
            # time.sleep(0.1)   # Used for debug
        self.setJointsConfiguration(current_config)
        return np.hstack((ee_pos, ee_orn))

    def compute_inverse_kin(self, end_effector_poses):
        """
        Computes a joint configuration solution corresponding to a desired end-effector pose. It uses PyBullet's default IK solver (damped least squres).
        Parameters:
            end_effector_poses: A numpy matrix with a desired end-effector pose for each row.
        Returns:
            joint_configurations: A list of numpy matrices containing the corresponding joint configurations computed via PyBullet IK
        """
        joint_configurations = np.nan * np.zeros((end_effector_poses.shape[0],
                                                  self.num_non_fixed_joints))
        for idx, ee_pose in tqdm(enumerate(end_effector_poses)):

            ee_target_pos_curr = ee_pose[:3].tolist()
            ee_target_orn_curr = ee_pose[3:].tolist()

            joint_configurations[idx, :] = self.bullet_client.calculateInverseKinematics(
                self.panda_id,
                panda_end_effector_index,
                ee_target_pos_curr,
                self.bullet_client.getQuaternionFromEuler(ee_target_orn_curr),
                self.lower_joint_limits,
                self.upper_joint_limits,
                self.joint_ranges,
                self.rest_configuration,
                maxNumIterations=20000,
                residualThreshold=0.00001)

        return joint_configurations[:, :self.num_controlled_joints]

    def getJointStates(self, robot_id):
        joint_states = self.bullet_client.getJointStates(robot_id, range(self.bullet_client.getNumJoints(robot_id)))
        joint_positions = [state[0] for state in joint_states]
        return joint_positions


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
            self.bullet_client.resetJointState(self.panda_id,
                                               j,
                                               desired_joints_configuration[j])

        attained_joints_configuration = desired_joints_configuration

        return attained_joints_configuration

    def getJointsConfiguration(self):
        """
        Returns the current joints configuration.

        Returns:
            joints_configuration: A Python list containing the current joint values
        """

        self.joints_configuration = []
        for j in range(self.num_non_fixed_joints):
            # Get joint limits from URDF
            jointState = self.bullet_client.getJointState(self.panda_id, j)
            self.joints_configuration.append(jointState[0])

        return self.joints_configuration


def generate_trajectory(ori, num_points):
    '''
    :param ori:
    :param num_points:
    :return:
    '''
    ee_trajectory_3D = iku3d.generate_circumference(center=[0.0, 0.044, -0.55],
                                                    radius=0.03,
                                                    num_points=num_points)  # number of sampled trajectory points
    ee_target_orientation = np.array([np.pi / 2., 0., ori])
    return np.hstack((ee_trajectory_3D, np.tile(ee_target_orientation, (num_points, 1))))
