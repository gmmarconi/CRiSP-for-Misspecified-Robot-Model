import sys
sys.path.extend(['.'])
import os
import seaborn; seaborn.set()
from pathlib import Path
import numpy as np
import pickle
from crisp.algorithms.NN import fc_nn
from crisp.utils.planar_utils import print_avg_distances_on_dataset
from crisp.utils.planar_utils import predict_panda_trajectory_CRiSP, predict_panda_trajectory_PB
from crisp.utils.data_utils import set_plt_params, set_logger
from crisp.algorithms.CRiSPIK import CRiSPIK
from crisp.algorithms.OneClassSSVM import OneClassSSVM
from crisp.algorithms.NN import trainer
import crisp.robots.panda_forward_kinematics as panda_fk
from crisp.robots.panda_urdf_randomizer import generate_custom_urdf
import datetime; time_format = '%a%d-%H%M%S'
import configparser
import argparse
import tempfile
import torch
import crisp.utils.panda_utils as iku3d
import itertools
import pybullet as p
import pybullet_data as pd
from numpy.random import default_rng

# Rescale EE positions
def psi(x):
    return np.hstack((x[:, :3]*20,
                      x[:, 3:]))

pos_dimensions = 3
save_alpha = False
save_Kx = False

timestring = datetime.datetime.now().strftime(time_format)

# ----Read configuration file and set-up logging---- #

#%% Read config files
parser = argparse.ArgumentParser(description='Train euclidean on toy dataset')
parser.add_argument('-config', help='Path to .ini config file')
args = parser.parse_args()
opt = parser.parse_args()

config = configparser.ConfigParser(allow_no_value=True)
assert len(config.read(opt.config)) > 0, "Couldn't load configuration file"
log = set_logger(config, timestring)

#%% Set seed and GPUs
if config.has_option('Experiment', 'seed'):
    rng = default_rng(config['Experiment'].getint('seed'))
    torch.manual_seed(config['Experiment'].getint('seed'))
else:
    config['Experiment']['seed'] = 77
    rng = default_rng(77)
    torch.manual_seed(77)
if config.has_option('Experiment', 'gpu'):
    device = torch.device("cuda:" + config['Experiment']['gpu'])
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

set_plt_params()

#%% ------Parse config parameters------ #
samples = config['Experiment'].getint('samples')
alg = config['Experiment']['algorithm']
output_folder = Path(config['Outputs']['output_folder'])
output_folder.mkdir(exist_ok=True)

if config.has_option('Experiment', 'orientation'):
    orientation = config['Experiment']['orientation']
else:
    orientation = True

if config.has_option('Experiment', 'reps'):
    reps = config['Experiment'].getint('reps')
else:
    reps = 1

if config.has_option('Alg Params', 's_search') or config.has_option('Alg Params', 'v_search'):
    try:
        s_list = [float(s.strip()) for s in config['Alg Params']['s_search'].split(",")]
    except KeyError:
        s_list = [float(s.strip()) for s in config['Alg Params']['s'].split(",")]
    try:
        v_list = [float(v.strip()) for v in config['Alg Params']['v_search'].split(",")]
    except KeyError:
        v_list = [float(v.strip()) for v in config['Alg Params']['v'].split(",")]

    hyperparams = list(itertools.product(s_list, v_list))
    reps = max(reps, len(hyperparams))
else:
    hyperparams = [(config['Alg Params'].getfloat('s'), config['Alg Params'].getfloat('v'))]

if config.has_option('Robot', 'lower_limits'):
    lower_limits = [ float(chunk.strip()) for chunk in config['Robot']['lower_limits'].split(",")]
if config.has_option('Robot', 'upper_limits'):
    upper_limits = [float(chunk.strip()) for chunk in config['Robot']['upper_limits'].split(",")]

boundaries = [(l,u) for l,u in zip(lower_limits, upper_limits)]

if config.has_option('Alg Params', 'loss_structure'):
    loss_structure = config['Alg Params']['loss_structure']
else:
    print(f"Please provide a loss structure type")
    sys.exit()

krls_flag = config['Experiment'].getboolean('krls')
falkon_flag = config['Alg Params'].getboolean('falkon')
train_flag = config['Experiment'].getboolean('train')
plot_hist_flag = config['Data'].getboolean('histograms')
save_model_flag = config['Outputs'].getboolean('save_model')

#%% ------Create/load Dataset------#

log.info("Loading dataset")
dataset = pickle.load(open(config['Data']['dataset'], 'rb'))

xtr = dataset['xtr']
ytr = dataset['ytr']
xval = dataset['xval']
yval = dataset['yval']

if config['Experiment'].getboolean('reconstruct_trajectory'):
    
    xte = {}

    results_CRiSP = {}
    results_PB = {}

    # 3D Spiral - Dataset: Medium Cube v0.2
    orn = 0	# Yaw orientation
    num_points = 200

    ee_trajectory_3D = iku3d.generate_spiral(center=[0.0, 0.044, -0.45],  # Center of the v0.2 cube
                         radius=0.03,  # spiral radius
                         theta_max=360 * 6,  # maximum angular displacement about spiral axis
                         height=-0.06,  # spiral height along axis
                         num_points=num_points)  # number of sampled trajectory points
    ee_target_orientation = np.array([np.pi / 2., 0., orn])
    xte['spiral_zero'] = np.hstack((ee_trajectory_3D, np.tile(ee_target_orientation, (num_points, 1))))

else:
    xte = dataset['xte']
    yte = dataset['yte']

if config['Experiment'].getboolean('use_validation_as_training') and alg != 'NN':
    log.info("Merging validation and training set...")
    xtr = np.vstack((xtr, dataset['xval']))
    ytr = np.vstack((ytr, dataset['yval']))

if config['Experiment'].getboolean('use_test_as_training'):
    log.info("Merging test and training set...")
    xtr = np.vstack((xtr, dataset['xte']))
    ytr = np.vstack((ytr, dataset['yte']))

if config['Data'].getboolean('preprocess'):
    if config['Model'].getboolean('preprocess'):
        CRiSPIK.preprocess(xtr)
        CRiSPIK.preprocess(xte)
        if xval is not None:
            CRiSPIK.preprocess(xval)

if config['Data'].getboolean('compute_distance_statistics'):
    log.info(f"Computing distances statistics...\n")
    print_avg_distances_on_dataset(psi(xtr), psi(xtr),
                                   dimensions=pos_dimensions,
                                   output_folder=output_folder,
                                   savename=f"hists_xtr={xtr.shape[0]}pts",
                                   plot_hist=plot_hist_flag,
                                   out=log.info)
log.info(f"Loaded dataset: {xtr.shape[0]} training points")

#%% ---------Initiaize pybullet sim------------ #
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pd.getDataPath())
timeStep = 1./60.
p.setTimeStep(timeStep)
p.setGravity(0, 0, 0)
num_sim_steps = 5000
rest_configuration = [0., 1.5708, 0., 4.7124, 0.]
base_orientation = [-0.707107, 0.0, 0.0, 0.707107]

# Instantiate simulated Panda object with
base_offset = [0, 0, 0]		# Base position offset (meters)
use_simulation = False		# Do not simulate the physics of the step or force new configuration

MMAE_orientation, MMAE_position = [], []
if isinstance(xte, dict):
    best = {items[0]: {'rmse': np.inf, 'hyperparams':(np.inf, np.inf), 'results':{}} for items in xte.items()}
else:
    best = {'rmse': np.inf, 'hyperparams':(np.inf, np.inf), 'results': {}}
struct_errors = {'RMSE_orientation': [], 'RMSE_position': [], 'RMSE': [], 'pointwise_RMSE': []}

for rep in range(reps):
    log.info(f"\n#####################################################\n"
             f"#------------------Repetition {rep+1}/{reps}-----------------#\n"
             f"#####################################################\n")
    bias_deg = config['Experiment'].getfloat('bias_deg')
    joint_bias = np.ones(7) * np.deg2rad(bias_deg)
    joint_bias *= np.sign(rng.random(7) - 0.5)
    joint_bias = tuple(joint_bias)

    bias_length = config['Experiment'].getfloat('bias_cm') * 1e-2
    link_bias = np.ones((9,3)) * bias_length
    link_bias *= np.sign(rng.random((9,3)) - 0.5)
    link_bias = tuple(map(tuple, link_bias))
    handle, model_file = tempfile.mkstemp(dir='./crisp/robots/urdf')

    with open(handle, 'w') as f:
        f.write(generate_custom_urdf(joint_bias, link_bias))

    # Prepare the pandasim objects
    pandasim_biased = panda_fk.PandaSimForwardKin(bullet_client=p,
                                                               base_offset=base_offset,
                                                               base_orientation=base_orientation,
                                                               urdf_path=model_file)
    pandasim = panda_fk.PandaSimForwardKin(bullet_client=p,
                                                        base_offset=base_offset,
                                                        base_orientation=base_orientation)
    if alg == 'CRiSP':
        s = hyperparams[rep][0]
        v = hyperparams[rep][1]

        if config.has_option('Alg Params', 'name'):
            save_name = config['Alg Params']['name']
        else:
            save_name=f'CRiSP_PANDA_{samples}pts.pickle'

        forward_biased = pandasim_biased.compute_forward_kin

        model = CRiSPIK(boundaries=boundaries,
                        s=s,
                        v=v,
                        psi=psi,
                        forward=forward_biased,
                        loss_structure=loss_structure,
                        krls=krls_flag,
                        use_leverage=config['Alg Params'].getboolean('leverage_scores'),
                        jacobian=None,
                        pos_dimensionality=3)

        if train_flag:
        # Train new model

            if xtr.shape[0] > 5000:
                while "The answer is invalid":
                    res = str(input("WARNING: The number of training points is n = " + str(xtr.shape[0]) + ". Training will require " + str(xtr.shape[0] ** 2 * 8) + " Bytes in RAM to compute the kernel matrix. Do you want to continue? (Enter y/n) ")).lower().strip()
                    if res[:1] == 'y':
                        model.fit(X=xtr, y=ytr,
                                  falkon=falkon_flag,
                                  Xte=xte,
                                  leverage_scores=config['Alg Params'].getboolean('leverage_scores'),
                                  out=log.info,
                                  krls_flag=krls_flag)
                        print("Training...")
                        break

                    elif res[:1] == 'n':
                        print("Quitting...")
                        sys.exit()

                    else:
                        print("The answer is invalid")

            else:
                print("Training...")
                model.fit(X=xtr, y=ytr,
                          falkon=falkon_flag,
                          Xte=xte,
                          leverage_scores=config['Alg Params'].getboolean('leverage_scores'),
                          out=log.info,
                          krls_flag=krls_flag)

            if save_model_flag:
                model.save(output_folder, save_name)

        else:
            # Load saved model
            model.load_state(Path(config['Data']['model']))

    elif alg == 'OC_SVM':
        biased_model = None
        s = hyperparams[rep][0]
        v = hyperparams[rep][1]
        if config.has_option('Alg Params', 'name'):
            save_name = config['Alg Params']['name']
        else:
            save_name=f'OC_SVM_PANDA_{samples}pts.pickle'

        model = OneClassSSVM(s=s, balls=True)

        if train_flag:
            model.fit(X=xtr, y=ytr)
            
            if save_model_flag:
                model.save(output_folder, save_name)
        else:
            model.load_state(Path(config['Data']['model']))

    elif alg == 'NN':
        biased_model = None
        max_epochs = config['Alg Params'].getint('max_epochs')
        lr = config['Alg Params'].getfloat('lr')
        if config.has_option('Alg Params', 'name'):
            save_name = config['Alg Params']['name']
        elif save_model_flag:
            save_name=f'NN_PANDA_{samples}pts.pickle'

        model = fc_nn(input_dim=xtr.shape[1], output_dim=ytr.shape[1], device=device).to(device)

        if next(model.parameters()).is_cuda:
            log.info("Neural network is on GPU")
        else:
            log.info("Neural network is on CPU")

        if train_flag:
            last, model = trainer(xtr, ytr,
                            model=model, device=device,
                            max_epochs=max_epochs, output_folder=output_folder, save_name=save_name,
                            xval=xval, yval=yval)
        else:
            model.load_state_dict(torch.load(Path(config['Data']['model']), map_location=device))
            # model = model.to(device)


    # %%------------- Predict trajectory and save data for best model----------##

    forward = pandasim.compute_forward_kin
    if isinstance(xte, dict):
        for traj_name, traj in xte.items():
            log.info(f"#----- Trajectory {traj_name} -----#")
            results_CRiSP[traj_name] = predict_panda_trajectory_CRiSP(traj, model.predict, forward, log.info, s, v, alg)
            if not save_alpha:
                results_CRiSP[traj_name]['alpha'] = np.array([])
            if not save_Kx:
                results_CRiSP[traj_name]['Kx'] = np.array([])
            if best[traj_name]['rmse'] > results_CRiSP[traj_name]['rmse']:
                best[traj_name]['rmse'] =  results_CRiSP[traj_name]['rmse']
                best[traj_name]['hyperparams'] = (s,v)
                best[traj_name]['results'] = results_CRiSP[traj_name]

        ## Compute ik with PyBullet just at the first repetition (it's deterministic)
        if rep == 0:
            for traj_name, traj in xte.items():
                log.info(f"#----- Trajectory {traj_name} -----#")
                results_PB[traj_name] = predict_panda_trajectory_PB(traj, pandasim_biased.compute_inverse_kin, forward, log.info, s, v, alg)
                if not save_alpha:
                    results_PB[traj_name]['alpha'] = np.array([])
                if not save_Kx:
                    results_PB[traj_name]['Kx'] = np.array([])

    else:
        results_CRiSP = predict_panda_trajectory_CRiSP(xte, model.predict, forward, log.info, s, v, alg)
        if best['rmse'] > results_CRiSP['rmse']:
            best['rmse'] = results_CRiSP['rmse']
            best['hyperparams'] = (s, v)
            best['results'] = results_CRiSP

        ## Compute ik with PyBullet just at the first repetition (it's deterministic)
        if rep == 0:
            for traj_name, traj in xte.items():
                log.info(f"#----- Trajectory {traj_name} -----#")
                results_PB = predict_panda_trajectory_PB(traj, pandasim_biased.compute_inverse_kin, forward, log.info, s, v, alg)
                if not save_alpha:
                    results_PB['alpha'] = np.array([])
                if not save_Kx:
                    results_PB['Kx'] = np.array([])

# Save results, including predicted trajectories
np.savez(str(output_folder) + "/results_CRiSP.npz", results_CRiSP)

os.remove(model_file)

log.info("\n#=================================================================#\n")
if isinstance(xte, dict):
    log.info("Best parameters for every trajectory:")
    for traj_name, trajs in xte.items():
        log.info(f"Best RMSE for {traj_name}: {best[traj_name]['rmse']:8.7f} with s: {best[traj_name]['hyperparams'][0]} and v: {best[traj_name]['hyperparams'][1]}\n")
        log.info(f"\t\t[PB] RMSE ori: {results_PB[traj_name]['rmse_orientation']:7.6f} ± {results_PB[traj_name]['var_orientation']:7.6f}")
        log.info(f"\t\t[PB] RMSE pos: {results_PB[traj_name]['rmse_position']:7.6f} ± {results_PB[traj_name]['var_position']:7.6f}\n"
                 f"\t\t[PB] RMSE: {results_PB[traj_name]['rmse']:7.6f} ± {results_PB[traj_name]['var']:7.6f}")

else:
    log.info(f"Best RMSE: {best['rmse']:8.7f} with s: {best['hyperparams'][0]} and v: {best['hyperparams'][1]}\n")

    log.info(f"\nAlgorithm: PyBullet Inverse Kinematics")
    log.info(f"\t\t[PB] RMSE ori: {results_PB['rmse_orientation']:7.6f} ± {results_PB['var_orientation']:7.6f}")
    log.info(f"\t\t[PB] RMSE pos: {results_PB['rmse_position']:7.6f} ± {results_PB['var_position']:7.6f}\n"
             f"\t\t[PB] RMSE: {results_PB['rmse']:7.6f} ± {results_PB['var']:7.6f}")
