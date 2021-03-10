# Structured Kinematics

Repository for:

_Gian Maria Marconi*, Raffaello Camoriano*, Lorenzo Rosasco, and Carlo Ciliberto, "Structured  Prediction  for  CRiSP  Inverse  Kinematics  Learning with  Misspecified  Robot  Models", IEEE Robotics and Automation Letters (RA-L) and IEEE International Conference on Robotics and Automation (ICRA) 2021 (to appear)_; Pre-print: https://arxiv.org/abs/2102.12942

>**Abstract**
>
> With the recent advances in machine learning, problems that traditionally would require accurate modeling to be solved analytically can now be successfully approached with data-driven strategies. Among these, computing the inverse kinematics of a redundant robot arm poses a significant challenge due to the non-linear structure of the robot, the hard joint constraints and the non-invertible kinematics map. Moreover, most learning algorithms consider a completely data-driven approach, while often useful information on the structure of the robot is available and should be positively exploited.
In this work, we present a simple, yet effective, approach for learning the inverse kinematics. We introduce a structured prediction algorithm that combines a data-driven strategy with the model provided by a forward kinematics function -- even when this function is inaccurate -- to efficiently tackle the problem. The proposed approach ensures that predicted joint configurations are well within the robot's constraints. We also provide statistical guarantees on the generalization properties of our estimator as well as an empirical evaluation of its performance on trajectory reconstruction tasks.


## Environment set-up

Tested on Ubuntu 18.04 and 20.04

We recommend using Anaconda for managing your environments

```
$ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
$ chmod +x Anaconda3-2020.07-Linux-x86_64.sh
$ ./Anaconda3-2020.07-Linux-x86_64.sh
```

Note: Specify a custom directory for Anaconda with enough disk space.

To install the Anaconda environment for this project, run in the shell

`$ conda env create -f environment.yml`

activate it with

`$ conda activate CRiSP-ik`

and verify it is correctly installed

`$ conda env list`

## How to run sample experiments

Included in this repo there are two sample scripts for experiments on the PANDA
robot or on the planar manipulator.
Both scripts read a configuration file (two samples of which are in the
  corresponding folder) and perform the experiment accordingly.

To launch the script, change directory to `experiments_scripts` and use:

- Planar maniputator: `$ python planar_manipulator_demo.py -config path/to/config.ini`
- Panda maniputator: `$ python panda_demo.py -config path/to/config.ini`

Where `config.ini` is a configuration file defining the experiment details.

### Rendering pre-computed Panda trajectory

Reproducing the Panda arm experiments requires 9.8GB of RAM for storing the full kernel matrix.
For convenience, we also provide a CRiSP-precomputed predicted trajectory in `experiments_scripts/outputs/panda_medium_cube/results_CRiSP.npz`
and a script for rendering the result in `PyBullet` in `experiments_scripts/render_panda_results.py`.

To render the predicted trajectory, change directory to `experiments_scripts` and run:

```
$ python render_panda_results.py
```

### Configuration file

The configuration file is in plaintext and has 4 sections:
  - [Experiment] Contains some generic infors on the experiment
  - [Trajectory] Specifies information on the trajectory parameters
  - [Data] Paths to datasets, pre-trained models, outputs etc.
  - [Alg Params] Parameters of the chosen algorithm/model

Boolean entries can be defined with either `0/no/false` or `1/yes/true`.


#### [Experiment]

  **Parameters**
- `reps`: The number of repetitions for the base experiment. This parameters is overwritten by the number of hyperparameters in grid_search mode
- `seed`: The random seed for the experiment
- `gpu`: The numerical Id of the GPU for deep learning based algorithms. If not present, defaults to CPU
- `algorithm`: Selected algorithm:
  - CRiSP (structured approach)
  - OCSSVM (One class SVM)
  - NN (Neural Network)
- `biad_cm`: The amount of bias in cm in every link of the robot (random sign)
- `bias_deg`: The amount of bias in degrees in every joint of the robot (random sign)
- `train`: if absent, the user must specify a pre-trained model in `[Data]`.
- `reconstruct_trajectory`: test on trajectory reconstruction (otherwise the mnodel is just trained)

#### [Outputs]
- `log_filename`: Path to a log file where the scripts outputs informations
- `output_folder`: folder where the scripts saves the results of the experiments

##### Only for `planar manipulator demo`
- `plot_dataset`: if `yes`, plots the generated dataset in `output_folder`
- `make_gif`: makes a gif of the reconstructed trajectory

#### [Trajectory]
- `type`: Type: either `eight` or `circle`
- `center_x`: x coordinate of the center of the trajectory in the workspace
- `center_y`: y coordinate of the center of the trajectory in the workspace
- `points`: number of points in the trajectory
- `scale`: a scale factor for the trajetory size. 1 is unscaled.


#### [Data]
- `generate_dataset`: is `no`, the script expects the path in `dataset`
- `dataset`: path to `.pickle` datasets
- `model`: model to load, ignored it `train` is `yes`
- `compute_distance_statistics`: if `yes`, the script computes some statistics on the training data inter-distances; can be slow for larger datasets
- `local`: if `yes`, the dataset is generated locally to the test Trajectory
- `max_dist`: how far from the trajectory is the neighbourhood of the dataset when `local` is `true`
- `preprocess`: if `yes` applies a preprocessing function to the training set

#### [Alg Params]
- `loss_structure`: Type of loss:
  - Forward: cartesian loss using the forward Kinematics
  - Radians: angular loss on the joints
- `name`: name with which the model will be saved in `output_folder`

##### when the selected `algorithm` is CRiSP
 - `s`: sigma of the Gaussian kernel
 - `v`: regularization parameter
 - `s_search`: a list of sigma values to evaluate the CRiSP on all combinations of s and v
 - `v_search`: a list of lambda values to evaluate the CRiSP on all combinations of s and v

##### when the selected `algorithm` is OCSSVM
- `s`: sigma of the Gaussian kernel
- `v`: regularization parameter
- `s_search`: a list of sigma values to evaluate the CRiSP on all combinations of s and v
- `v_search`: a list of lambda values to evaluate the CRiSP on all combinations of s and v

##### when the selected `algorithm` is NN
- `max_epochs`: number of epochs for training the NN
- `lr`: learning rate of Adam optimizer
