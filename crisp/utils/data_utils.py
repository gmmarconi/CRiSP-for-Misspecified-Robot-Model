from pathlib import Path
import imageio
import matplotlib.pyplot as plt
import logging


def make_gif(gif_folder, save_name=None):
    """ Collects numbered images from folder called "giffy" and makes a gif with save_name in output_folder """
    images_buffer = []
    for filename in sorted([f for f in (gif_folder).glob('**/*') if f.is_file()]):
        if filename.suffix == '.png':
            images_buffer.append(imageio.imread(filename))
    if save_name is None:
        save_name = 'reconstructed_trajectory'
    imageio.mimwrite(gif_folder.parent / (save_name+'.gif'), images_buffer, fps=7) #, format='GIF-FI')


def set_plt_params():
    """    Sets pyplot parameters    """
    plt.rcParams['figure.figsize'] = [12, 9]
    plt.rcParams['lines.markersize'] = 8
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['xtick.labelsize'] = 25
    plt.rcParams['ytick.labelsize'] = 25
    plt.rcParams['legend.fontsize'] = 20

def set_logger(config, id_string=None):
    """
    Sets up and returns a logger object. Using "log.info"
    outputs text to both shell and logfile.
    :param config: config object from configparser
    :param id_string: identificative string for logfile
    :return: log object
    """

    if config.has_option('Outputs', 'log_filename') and config.has_option('Outputs', 'output_folder'):
        filename = Path(config['Outputs']['log_filename']).stem + \
                  "_" + id_string + \
                  Path(config['Outputs']['log_filename']).suffix
        logfile = Path(config['Outputs']['output_folder']) / filename
    else:
        logfile = Path('logs/struct_kinematics_' + id_string + '.log')
    log = logging.getLogger('logger')
    log.setLevel(level=logging.INFO)
    chandler = logging.StreamHandler()
    chandler.setLevel(logging.INFO)
    log.addHandler(chandler)
    logfile.parent.mkdir(exist_ok=True, parents=True)
    fhandler = logging.FileHandler(logfile, mode='w')
    fhandler.setLevel(logging.INFO)
    log.addHandler(fhandler)

    # Print config file
    log.info("\n\nConfiguration file for experiment")
    for section in config.sections():
        log.info(f"\n{section}")
        for option in config.options(section):
            if config.get(section, option):
                log.info(f"{option} = {config.get(section, option)}")
            else:
                log.info(f"{option}")

    return log


def print_trajectory_results(alg, results, s, v, out):
    """Prints a formatted version of one repetition of trajectory reconstruction experiments"""

    out(f"Algorithm: {alg}")
    if alg in ['CRiSP', 'OC_SVM']:
        out(f"s: {s}\t v: {v}")
    if results['rmse_orientation']:
        out(f"\t\tRMSE ori: {results['rmse_orientation']:7.6f} ± {results['var_orientation']:7.6f}")
    out(f"\t\tRMSE pos: {results['rmse_position']:7.6f} ± {results['var_position']:7.6f}\n"
             f"\t\tRMSE: {results['rmse']:7.6f} ± {results['var']:7.6f}")

    if 'bias_error' in results:
        out("#Biased model results#")
        if results['bias_error']['rmse_orientation']:
            out(f"\t\tRMSE ori: {results['bias_error']['rmse_orientation']:7.6f} ± {results['bias_error']['var_orientation']:7.6f}")
        out(f"\t\tRMSE pos: {results['bias_error']['rmse_position']:7.6f} ± {results['bias_error']['var_position']:7.6f}\n"
            f"\t\tRMSE: {results['bias_error']['rmse']:7.6f} ± {results['bias_error']['var']:7.6f}\n")
    print("")