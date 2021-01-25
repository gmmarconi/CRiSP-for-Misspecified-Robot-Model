import sys
sys.path.extend(['..'])
import seaborn; seaborn.set()
from pathlib import Path
import numpy as np
import pickle
from crisp.algorithms.NN import fc_nn
from crisp.utils.planar_utils import generate_circle, generate_8, print_avg_distances_on_dataset, plot_dataset
from crisp.utils.data_utils import set_plt_params, set_logger, make_gif, print_trajectory_results
from crisp.robots.planar_5links_robot import planar_5links_robot
from crisp.robots.planar_urdf_randomizer import generate_custom_urdf
from crisp.algorithms.CRiSPIK import CRiSPIK
from crisp.algorithms.OneClassSSVM import OneClassSSVM
from crisp.algorithms.NN import trainer
import datetime; time_format = '%a%d-%H%M%S'
import configparser
import argparse
import torch
import tempfile
import itertools
from numpy.random import default_rng


def psi(x):
    """ Feature map """
    return np.hstack((x[:, :2] / 500,
                      0.5*(1 / (1 + np.exp(-x[:, :2]))) - 0.5,
                      np.atleast_2d(np.sin(x[:, 2])).T, np.atleast_2d(np.cos(x[:, 2])).T,
                      (np.atleast_2d(np.amin([np.abs(x[:, 2]), 2*np.pi-np.abs(x[:, 2])], axis=0)).T / (np.pi * 0.5)) - 1))
pos_dimensions = 6

# def psi(x):
#     return x
# pos_dimensions = 2
timestring = datetime.datetime.now().strftime(time_format)
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
link_length = config['Experiment'].getint('link_length')  # Not used, defined in code later
samples = config['Data'].getint('samples')
alg = config['Experiment']['algorithm']
output_folder = Path(config['Outputs']['output_folder'])
output_folder.mkdir(parents=True, exist_ok=True)
if config.has_option('Experiment', 'orientation'):
    orientation = config['Experiment']['orientation']
else:
    orientation = True

if config.has_option('Experiment', 'reps'):
    reps = config['Experiment'].getint('reps')
    s_list = [config['Alg Params'].getfloat('s')] * reps
    v_list = [config['Alg Params'].getfloat('v')] * reps
    hyperparams = list(itertools.product(s_list, v_list))

else:
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
        reps = len(hyperparams)
    else:
        hyperparams = [(config['Alg Params'].getfloat('s'), config['Alg Params'].getfloat('v'))]
        reps = 1

if config.has_option('Alg Params', 'loss_structure'):
    loss_structure = config['Alg Params']['loss_structure']
else:
    print(f"Please provide a loss structure type")
    sys.exit()

traj_num_pts = config['Trajectory'].getint('num_points')
falkon_flag = config['Alg Params'].getboolean('falkon')
train_flag = config['Experiment'].getboolean('train')
analyze_preds = config['Outputs'].getboolean('analyze_preds')
plot_hist_flag = config['Data'].getboolean('histograms')
validation_flag = config['Experiment'].getboolean('validate') or config['Experiment'].getboolean('validation')

#%% ------Generate trajectory------ #
traj_name = config['Trajectory']['type']
cx = config['Trajectory'].getfloat('center_x')
cy = config['Trajectory'].getfloat('center_y')
scale = config['Trajectory'].getfloat('scale')

if traj_name == 'circle':  #cx=1.7, cy=2  cx=3.4,m cy=3
    traj = generate_circle(cx=cx, cy=cy, r=scale, num_points=traj_num_pts)
    # traj2 = generate_8(cx=ecx, cy=ecy, num_points=100)
elif traj_name == 'eight': # cx=3, cy=2.5
    traj = generate_8(cx=cx, cy=cy, num_points=traj_num_pts, scale=scale, rotation=np.pi/2)
    # traj2 = generate_circle(cx=ccx, cy=ccy, r=1, num_points=100)

if config.has_option('Alg Params', 'loss_structure'):
    loss_structure = config['Alg Params']['loss_structure']
else:
    print(f"Please provide a loss structure type")
    sys.exit()

struct_errors = {'RMSE_orientation': [], 'RMSE_position': [], 'RMSE': [], 'pointwise_RMSE': []}
pb_errors = {'RMSE_orientation': [], 'RMSE_position': [], 'RMSE': [], 'pointwise_RMSE': []}
results_inv_pb = []
best = {'rmse': np.inf, 's': np.nan}
# pointwise_RMSE_struct = np.array((reps, traj_num_pts, 3))
# pointwise_RMSE_pb = np.array((reps, traj_num_pts, 3))

for rep in range(reps):
    log.info(f"\n#####################################################\n"
             f"#------------------Repetition {rep+1}/{reps}-----------------#\n"
             f"#####################################################")

    bias_deg = config['Experiment'].getfloat('bias_deg')
    joint_bias = np.ones(5) * np.deg2rad(bias_deg)
    joint_bias *= np.sign(rng.random(5) - 0.5)

    bias_length = config['Experiment'].getfloat('bias_cm') * 1e-2
    link_bias = np.ones(5) * bias_length
    link_bias *= np.sign(rng.random(5) - 0.5)

    handle_biased, urdf_biased = tempfile.mkstemp()
    with open(handle_biased, 'w') as f:
        f.write(generate_custom_urdf(joint_bias, link_bias))
    planar_manip_biased = planar_5links_robot(urdf_path=urdf_biased)

    handle_correct, urdf_correct = tempfile.mkstemp()
    with open(handle_biased, 'w') as f:
        f.write(generate_custom_urdf(joint_bias, link_bias))
    planar_manip = planar_5links_robot(urdf_path=urdf_correct)


    #%% ------Create/load Dataset------#
    if config['Data'].getboolean('generate_dataset'):
        log.info("Generating dataset")
        if config['Data'].getboolean('local'):
            if config.has_option('Data', 'max_dist'):
                max_dist = config['Data'].getfloat('max_dist')
            else:
                max_dist = 1
            dataset = planar_manip.generate_dset_near_trajectory(trajectory=traj,
                                                                 max_dist=max_dist,
                                                                 samples=samples,
                                                                 orientation=config['Data'].getboolean('orientation'),
                                                                 savename=Path(config['Data']['dataset']).name,
                                                                 save_folder=Path('datasets/synth'))
        else:
            dataset = planar_manip.generate_dset(samples=samples, save_folder=Path(config['Data']['dataset']).parent,
                                                 savename=Path(config['Data']['dataset']).name)

    elif config.has_option('Data', 'dataset') and not config['Data'].getboolean('generate_dataset'):
        log.info("Loading dataset")
        dataset = pickle.load(open(config['Data']['dataset'], 'rb'))

    if alg != 'NN':
        log.info(f"Joining training and validation")
        xtr = np.vstack((dataset['xtr'], dataset['xval']))
        ytr = np.vstack((dataset['ytr'], dataset['yval']))
        xval = None
    else:
        log.info(f"Loading separately training and validation")
        xtr = dataset['xtr']
        ytr = dataset['ytr']
        xval = dataset['xval']
        yval = dataset['yval']

    if validation_flag:
        xte = dataset['xval']
        yte = dataset['yval']
    else:
        xte = dataset['xte']
        yte = dataset['yte']

    log.info(f"Loaded dataset: {xtr.shape[0]} training points")

    if config['Data'].getboolean('preprocess'):
        if config['Model'].getboolean('preprocess'):
            CRiSPIK.preprocess(xtr)
            CRiSPIK.preprocess(xte)
            if xval is not None:
                CRiSPIK.preprocess(xval)

    #------Plot dataset------ #
    if config['Data'].getboolean('plot_dataset') and rep == 0:
        plot_dataset(xtr, ytr, traj, planar_manip, output_folder)

    if config['Data'].getboolean('compute_distance_statistics') and rep == 0:
        print_avg_distances_on_dataset(psi(xtr), psi(xtr),
                                       dimensions=pos_dimensions,
                                       output_folder=output_folder,
                                       savename=f"hists_xtr={xtr.shape[0]}pts",
                                       plot_hist=plot_hist_flag,
                                       out=log.info)

#%% ------Load algorithm-wise parameters and train or load model------ #
    if alg == 'CRiSP':
        biased_model = None
        s = hyperparams[rep][0]
        v = hyperparams[rep][1]
        if config.has_option('Alg Params', 'name'):
            save_name = config['Alg Params']['name']
        else:
            save_name=f'CRiSP_{traj_name}_{samples}pts.pickle'

        model = CRiSPIK(boundaries=planar_manip.get_boundaries(),
                        s=s,
                        v=v,
                        psi=psi,
                        forward=planar_manip_biased.computeForwardKinematics,
                        loss_structure=loss_structure,
                        pos_dimensionality=2,
                        use_leverage=config['Alg Params'].getboolean('leverage_scores'),
                        random_seed=config['Experiment'].getint('seed'))

        if train_flag:
            log.info(f"Beginning Training...")
            model.fit(X=xtr, y=ytr,
                      falkon=falkon_flag, Xte=traj,
                      out=log.info)
            model.save(output_folder, save_name)
        else:
            model.load_state(Path(config['Data']['model_to_load']))

    elif alg == 'OCSVM':
        biased_model = None
        # best: g=0.2, v=0.001
        s = hyperparams[rep][0]
        v = hyperparams[rep][1]
        if config.has_option('Alg Params', 'name'):
            save_name = config['Alg Params']['name']
        else:
            save_name=f'OCSVM_{traj_name}_{samples}pts.pickle'

        model = OneClassSSVM(s=s,
                             v=v,
                             balls=True,
                             boundaries=planar_manip.get_boundaries())

        if train_flag:
            model.fit(X=xtr, y=ytr)
        else:
            model.load_state(Path(config['Data']['model_to_load']))

    elif alg == 'NN':
        s = np.nan
        v = np.nan
        biased_model = None
        max_epochs = config['Alg Params'].getint('max_epochs')
        lr = config['Alg Params'].getfloat('lr')
        if config.has_option('Alg Params', 'name'):
            save_name = config['Alg Params']['name']
        else:
            save_name=f'NN_{traj_name}_{samples}pts.pickle'

        model = fc_nn(input_dim=xtr.shape[1], output_dim=ytr.shape[1], device=device).to(device)

        if next(model.parameters()).is_cuda:
            print("Neural network is on GPU")
        else:
            print("Neural network is on CPU")

        if train_flag:
            last, model = trainer(xtr, ytr,
                            model=model, device=device,
                            max_epochs=max_epochs, output_folder=output_folder, save_name=save_name,
                            xval=xval, yval=yval)
        else:
            model.load_state_dict(torch.load(Path(config['Data']['model_to_load']), map_location=device))

#%% --------------------Trajectory reconstruction-------------------- #
    if config['Experiment'].getboolean('reconstruct_trajectory'):
        print("Beginning trajectory reconstruction")
        if config.has_option('Experiment', 'bias_percentage') and config['Experiment'].getfloat('bias_percentage') != 0:
            plot_bias = True
        else:
            plot_bias=False

        results = planar_manip_biased.test_trajectory(true_kinematics_model=planar_manip,
                                                      trajectory=traj,
                                                      output_folder=output_folder,
                                                      traj_name=traj_name,
                                                      model=model,
                                                      analyze_preds=analyze_preds,
                                                      id_string=f"{timestring}",
                                                      plot_bias=plot_bias,
                                                      save_svg=False,
                                                      giffable=config['Outputs'].getboolean('make_gif'))

        results_inv_pb = planar_manip_biased.test_trajectory_inv_pb(true_kinematics_model=planar_manip,
                                                                    trajectory=traj,
                                                                    output_folder=output_folder,
                                                                    traj_name=traj_name,
                                                                    id_string=f"{timestring}",
                                                                    plot_bias=plot_bias)

        if config['Outputs'].getboolean('make_gif'):
            # make_gif(Path(f'alpha_plots/{timestring}_s{str(s)}'), output_folder, save_name=f"alphas_{timestring}_s{str(s)}")
            make_gif(Path(f'giffy'), output_folder, save_name=f"Reconstructed with {alg}- {timestring}")

        print_trajectory_results(alg, results, s, v, log.info)

        print("")
        print_trajectory_results("Inverse Kinematics Pybullet", results_inv_pb, s, v, log.info)

        results['traj'] = traj

        def store_errors(store_dict, results):
            store_dict['RMSE_orientation'].append(results['rmse_orientation'])
            store_dict['RMSE_position'].append(results['rmse_position'])
            store_dict['RMSE'].append(results['rmse'])

        if best['rmse'] > results['rmse']:
            best['rmse'] = results['rmse']
            best['hyperparams'] = (s, v)

        store_errors(struct_errors, results)
        store_errors(pb_errors, results_inv_pb)

    del planar_manip, model


#%% Final statistics output
log.info("\n#=================================================================#\n")
if config.has_option('Experiment', 'reps'):
    log.info(f"%----------------------{alg} Average Error----------------------%")
    log.info(f"Best RMSE: {best['rmse']:8.7f} with s: {best['hyperparams'][0]} and v: {best['hyperparams'][1]}\n")

    def print_final_stats(store_dict, log):
        if store_dict['RMSE_orientation'][0]:
            log.info(f"RMSE orientation: {np.mean(store_dict['RMSE_orientation']):8.7f} ± {np.std(store_dict['RMSE_orientation']):8.7f}")
        log.info(f"RMSE position: {np.mean(store_dict['RMSE_position']):8.7f} ± {np.std(store_dict['RMSE_position']):8.7f}\n"
                f"RMSE: {np.mean(store_dict['RMSE']):8.7f} ± {np.std(store_dict['RMSE']):8.7f}\n")

    print_final_stats(struct_errors, log)

    log.info("%----------------------PyBullet Error----------------------%")
    print_final_stats(pb_errors, log)
    log.info(f"%----------------------------------------------------------------%\n")


