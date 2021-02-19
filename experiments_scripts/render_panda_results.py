import sys
sys.path.extend(['.'])
import numpy as np
import warnings
import pybullet as p
import pybullet_data as pd
# import experiments.panda_sandbox.inverse_kin_utils_3D as iku3d
# import experiments.panda_sandbox.panda_sample_cartesian_traj_sim as panda_sample_cartesian_traj_sim
import time
from scipy.io import loadmat, savemat

# Panda config
panda_end_effector_index = 11 #8
panda_num_dofs = 7
base_orientation = [-0.707107, 0.0, 0.0, 0.707107]
base_offset = [0, 0, 0]		# Base position offset [0,0,0] (meters)

# General PyBullet config
gui = True
use_simulation = False

# Startup PyBullet
if gui:
    p.connect(p.GUI)
    # p.connect(p.GUI, options="--opengl2")
    p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
else:
    p.connect(p.DIRECT)

p.setAdditionalSearchPath(pd.getDataPath())
flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES

# Set simulation time step
timeStep = 1./60.
p.setTimeStep(timeStep)

# Set gravity
p.setGravity(0, -9.81, 0)

# Load trajectories

trajectories_fname = 'outputs/panda_medium_cube/results_CRiSP.npz'

# Name of the trajectory to be rendered
traj_name = 'spiral_zero'   # Default

trajectories = np.load(trajectories_fname, allow_pickle=True)
trajectories = trajectories.f.arr_0.item()

# Load Panda arm
panda = p.loadURDF("franka_panda/panda.urdf",
                     np.array([0, 0, 0]) + base_offset,
                     base_orientation,
                     useFixedBase=True,
                     flags=flags)

# Cycle on dataset examples

n = trajectories[traj_name]['trajectory_joints_pred_CRiSP'].shape[0]

trajectory_pos_ikpb = np.zeros((n, 3))
trajectory_pos_CRiSP = np.zeros((n, 3))
trajectory_target_pos = np.zeros((n,3))

trajectory_orn_ikpb = np.zeros((n, 3))
trajectory_orn_CRiSP = np.zeros((n, 3))
trajectory_target_orn = np.zeros((n,3))

for k in range(n):

    # Get k-th joints configuration and EE pose
    joints_configuration = trajectories[traj_name]['trajectory_joints_pred_CRiSP'][k, :]
    ee_pos_dset = trajectories[traj_name]['xte'][k, :3]
    ee_orn_dset = trajectories[traj_name]['xte'][k, 3:]

    # Force set joints to predicted configuration
    for i in range(panda_num_dofs):
        p.resetJointState(panda,
                          i,
                          joints_configuration[i])

    # Readout EE pose
    ee_state = p.getLinkState(panda, panda_end_effector_index, computeForwardKinematics=True)
    ee_pos_curr = np.array(ee_state[4])
    ee_orn_curr = np.array(p.getEulerFromQuaternion(np.array(ee_state[5]).tolist()))

    # Compare EE pose with stored one
    trajectory_pos_CRiSP[k] = ee_pos_curr
    trajectory_target_pos[k] = ee_pos_dset
    trajectory_orn_CRiSP[k] = ee_orn_curr
    trajectory_target_orn[k] = ee_orn_dset

    if gui:
        time.sleep(1./60.)

        # Plot difference between current and target EE pos
        p.addUserDebugLine(ee_pos_dset, ee_pos_curr, [1, 0, 0], lifeTime=50)

        if k > 0:
            # Plot desired trajectory
            p.addUserDebugLine(trajectory_target_pos[k-1], ee_pos_dset, [0, 1, 0], lifeTime=50)

            # Plot actual trajectory
            p.addUserDebugLine(trajectory_pos_CRiSP[k - 1], ee_pos_curr, [0, 0, 1], lifeTime=50)


