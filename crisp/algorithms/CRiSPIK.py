import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted
from crisp.utils.planar_utils import sqdist, sqdist_weighted, radial_squared_error
from scipy.optimize import minimize
from scipy.linalg import lapack
import time
import pickle
import sys
from tqdm import tqdm
import torch as th
import datetime
import matplotlib
import matplotlib.pyplot as plt


class CRiSPIK(BaseEstimator):
    """ Consistent Regularized Estimaator class.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """

    def __init__(self, pos_dimensionality,
                 psi=None, forward=None,
                 boundaries=None, s=None, v=None,
                 loss_structure=None, krls=False,
                 use_leverage=False,
                 jacobian=None,
                 random_seed=None):
        """Constructor of inverse kinematics structured estimator

        :param boundaries: List of duples for lower and upper joint boundaries
        :param g: gamma for Gaussian kernel
        :param loss_structure: "Euclidean" or "Spherical" for choosing the type of loss
        """
        self.is_fitted_ = False
        self.M_inv = None
        self.X = None
        self.Y = None
        if isinstance(s, list):
            self.s = np.array([float(s) for s in s])
        else:
            self.s = float(s)
        self.v = float(v)
        self.loss_structure = loss_structure
        self.boundaries = boundaries
        self.outdim = len(boundaries)
        self.krls_preds = None
        self.krls = krls
        self.Kx = None
        self.forward = forward
        self.pos_dimensionality = pos_dimensionality
        if pos_dimensionality == 2:
            self.f_cartesian = self.f_cartesian_2d
        elif pos_dimensionality == 3:
            self.f_cartesian = self.f_cartesian_3d
        if psi is not None:
            self.psi = psi
        if jacobian is not None:
            self.jacobian = jacobian
        self.leverage_scores = None
        self.use_leverage = use_leverage

        # print(f"Loss structure: {self.loss_structure}")
        self.rng = np.random.default_rng(random_seed)

    def fit(self, X, y, **kwargs):
        if 'out' not in kwargs:
            kwargs['out'] = None
        return self.fit_classic(X, y, kwargs['out'])

    def fit_classic(self, X, y, out=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """

        self.is_fitted_ = True
        self.X = X
        self.Y = y
        n = X.shape[0]

        if out is None:
            out = print

        out(f"Training CRiSP model with {n} training points")
        t0 = datetime.datetime.now()
        K = self.K_matrix(self.psi(self.X))
        self.M_inv = invPD(K + n * self.v * np.identity(n))
        t1 = datetime.datetime.now()
        out(f"Training completed in {str(datetime.timedelta(seconds=(t1 - t0).seconds))}\n")

        if self.use_leverage:
            # self.leverage_scores = np.diagonal(K @ self.M_inv)
            self.leverage_scores = np.einsum('ij,ji->i', K, self.M_inv)
        # `fit` should always return `self`
        return self

    def predict_krls(self, X, is_sequence=False, out=None):
        check_is_fitted(self, 'is_fitted_')
        xte = np.atleast_2d(X)
        if out is None:
            out = print
        if self.Kx is None:
            self.Kx = self.K_matrix(self.psi(self.X), self.psi(xte))
        alpha = np.dot(self.M_inv, self.Kx).T
        self.Kx = None
        return alpha @ self.Y, {}

    def predict(self,
                X,
                y=None,
                y0=None,
                is_sequence=False,
                out=None):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of ones.
        """
        check_is_fitted(self, 'is_fitted_')
        X = np.atleast_2d(X)
        xte = check_array(X, accept_sparse=False)
        y0s = []
        y_amax = []
        y_mean = []
        a_indexes = []
        data = {}
        if self.loss_structure == 'Forward':
            self.Ptr = self.forward(self.Y)

        if self.use_leverage:
            ## Lev scores
            print("Computing leverage scores")
            if self.leverage_scores is None:
                self.leverage_scores = np.einsum('ij,ji->i', self.K_matrix(self.psi(self.X)), self.M_inv)

            rng = np.random.default_rng(77)
            print("computed LS")
            leverage_sampled = np.full(self.X.shape[0], np.inf)
        if out is None:
            out = print

        # compute alphas
        self.Kx = self.K_matrix(self.psi(self.X), self.psi(xte))
        # self.Kx = np.exp(-sqdist(self.psi(self.X), self.psi(xte)) / (2*self.s**2))
        alpha = np.atleast_2d(np.dot(self.M_inv, self.Kx).T)

        y_hat = np.full((X.shape[0], self.outdim), np.inf)
        if X.shape[0] > 1:
            out(f"Predicting {X.shape[0]} test points.\tLoss structure: {self.loss_structure}")
            t0 = datetime.datetime.now()

        for idx, alpha_x in (enumerate(tqdm(alpha)) if X.shape[0] > 1 else enumerate(alpha)):
            idx_a = np.arange(len(alpha_x))
            a_indexes.append(idx_a)
            a_x = np.ndarray.copy(alpha_x[idx_a])

            if y0 is None and (not is_sequence or idx == 0):
                y0 = np.ndarray.copy(self.Y[np.argmax(a_x)])
                # y0 = np.array([(high-low)*np.random.random() + low for low, high in self.boundaries])
            elif is_sequence and idx > 0:
                y0 = np.ndarray.copy(y_hat[idx - 1])

            y0s.append(y0)
            y_amax.append(np.ndarray.copy(self.Y[np.argmax(alpha_x)]))
            if self.loss_structure == 'Radians':
                y_hat[idx] = minimize(self.f_circle, x0=y0, args=(a_x, self.Y[idx_a]),
                                      method='L-BFGS-B',
                                      bounds=self.boundaries,
                                      jac=self.grad_f_circle,
                                      options={'gtol': 1e-6, 'disp': False}).x[np.newaxis, :]
            elif self.loss_structure == 'Forward':
                y_hat[idx] = np.atleast_2d(minimize(self.f_cartesian, x0=y0, args=(a_x, idx_a), method='L-BFGS-B',
                                                    bounds=self.boundaries,
                                                    # jac='2-point',
                                                    options={'gtol': 1e-12,
                                                             'ftol': 1e-12,
                                                             'maxls': 200,
                                                             'eps': 1e-4,
                                                             'disp': False}).x)
            else:
                out("Please supply a valid mode: Cartesian or Radians")
            y0 = None

            if not self.inside_boundaries(y_hat[idx]):
                out(f"[{idx}] outside of boundaries")
        if X.shape[0] > 1:
            t1 = datetime.datetime.now()
            out(f"Inference completed in {str(datetime.timedelta(seconds=(t1 - t0).seconds))} mins")

        if self.use_leverage:
            print(f"Average samples: {np.mean(leverage_sampled)} ± {np.std(leverage_sampled):.2f}")
        data['alpha'] = alpha
        data['Kx'] = self.Kx
        if y0s: data['y0s'] = y0s
        if y_amax: data['y_amax'] = y_amax
        if y_mean: data['y_mean'] = y_mean
        if a_indexes: data['a_indexes'] = a_indexes
        self.Ptr = None
        self.Kx = None

        return y_hat, data



    #### Radians Loss #####
    def f_circle(self, x, alpha, ytr):
        """ Functional to be minimized for Consistent Regularized Structured Prediction.
        Reference:
        Ciliberto, Carlo, Lorenzo Rosasco, and Alessandro Rudi.
        "A consistent regularization approach for structured prediction."
        Advances in neural information processing systems. 2016.."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return np.dot(alpha, self.sq_delta_circle(x, ytr))

    def sq_delta_circle(self, y, ytr):
        """Computes distance as the squared sum of radians distances"""
        if y.ndim == 1:
            y = y[np.newaxis, :]
        return np.sum(np.amin([np.abs(y - ytr), 2 * np.pi - np.abs(y - ytr)], axis=0) ** 2, axis=1)

    def grad_f_circle(self, x, alpha, ytr):
        """Gradient of the structured loss on the circle"""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        return np.dot(alpha, self.grad_sq_delta_circle(x, ytr)) / 2

    def grad_sq_delta_circle(self, y, ytr):
        """Gradient of the sum of squared radians distances"""
        if y.ndim == 1:
            y = y[np.newaxis, :]
        delta = y - ytr
        return -np.amin([np.abs(delta), 2 * np.pi - np.abs(delta)], axis=0) * np.sign(delta) * np.sign(
            np.abs(delta) - np.pi)

    ### Forwards Loss ###
    def f_cartesian_2d(self, x, alpha, idx_a):
        """Functional to be minimized for Consistent Regularized Structured Prediction.
        Uses the forward kinematics in self.for4ward to compute the Euclidean loss in the workspace
        Reference:
        Ciliberto, Carlo, Lorenzo Rosasco, and Alessandro Rudi.
        "A consistent regularization approach for structured prediction."
        Advances in neural information processing systems. 2016.

        :param x: candidate configuration point (set of joint angles)
        :param alpha: alphas associated with the training points
        :param ytr: configuration training points
        :return: a single scalar loss
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        P = self.forward(x)

        D_Euclidean = sqdist(P[:, :self.pos_dimensionality], self.Ptr[idx_a, :self.pos_dimensionality])
        D_angular = np.amin([np.abs(P[:, 2] - self.Ptr[idx_a, 2]), 2 * np.pi - np.abs(P[:, 2] - self.Ptr[idx_a, 2])],
                            axis=0) ** 2

        return np.dot(10 * D_Euclidean + D_angular, alpha)
        # return np.dot(D_Euclidean, alpha)

    ### Forwards Loss ###
    def f_cartesian_3d(self, x, alpha, idx_a):
        """Functional to be minimized for Consistent Regularized Structured Prediction.
        Uses the forward kinematics in self.for4ward to compute the Euclidean loss in the workspace
        Reference:
        Ciliberto, Carlo, Lorenzo Rosasco, and Alessandro Rudi.
        "A consistent regularization approach for structured prediction."
        Advances in neural information processing systems. 2016.

        :param x: candidate configuration point (set of joint angles)
        :param alpha: alphas associated with the training points
        :param ytr: configuration training points
        :return: a single scalar loss
        """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        P = self.forward(x)

        D_Euclidean = self.delta_Euclidean(P[:, :3], self.Ptr[idx_a, :3])
        D_angular = self.sq_delta_circle(P[:, 3:], self.Ptr[idx_a, 3:])
        return np.dot(D_Euclidean + D_angular, alpha)

    def save(self, output_folder, save_name):
        """
        Saves the model in a .pickle file
        :param output_folder: destination folder
        :param save_name: name of the file
        """
        CRiSP = {'M_inv': self.M_inv, 'X': self.X, 'Y': self.Y, 's': self.s, 'psi': self.psi}
        print(
            f"Saving model in {output_folder}...\tKernel matrix inverse size: \t{sys.getsizeof(CRiSP['M_inv']) * 1e-9:3f}GB")

        if save_name is None:
            save_name = 'CRiSP'
        pickle.dump(CRiSP, open(output_folder / (save_name + ".pickle"), 'wb'), protocol=4)
        print(f"Save complete!")

    def load_state(self, path):
        """ Loads a model saved with 'save' function """
        try:
            state = pickle.load(open(path, 'rb'))
        except FileNotFoundError:
            print(f"CRiSP_tjm.load_state(): Could not find the model to load, please check the path")
            sys.exit()
        self.M_inv = state['M_inv']
        self.s = state['s']
        self.X = state['X']
        self.Y = state['Y']
        self.psi = state['psi']
        self.is_fitted_ = True

    def inside_boundaries(self, Q):
        for q, bounds in zip(Q, self.boundaries):
            if q < bounds[0] or q > bounds[1]:
                return False
            return True

    def K_matrix(self, X1, X2=None, verbose=False):
        """Compute the kernel matrix of a Gaussian kernel given two matrices of points"""
        if X2 is None:
            X2 = X1
        if isinstance(self.s, float) or isinstance(self.s, int):
            sqD = sqdist(X1, X2)
        elif isinstance(self.s, list) or isinstance(self.s, np.ndarray):
            sqD = sqdist_weighted(X1, X2, self.s)
        if verbose:
            D = np.sqrt(sqD)
            D_sorted = np.sort(D, axis=1)
            print(f"Avg distance 5th nearest neighbour: {np.mean(D_sorted[:, 5]):.3f}  ± {np.std(D_sorted[:, 5]):.3f}\n"
                  f"Avg distance 10th nearest neighbour: {np.mean(D_sorted[:, 10]):.3f}  ± {np.std(D_sorted[:, 10]):.3f}\n"
                  f"Mean distance: {np.mean(D[D > 0]):.3f} ± {np.std(D[D > 0]):.3f}\n"
                  f"Avg Min distance: {np.mean(D_sorted[:, 1]):.3f} ± {np.std(D_sorted[:, 1]):.3f}\n"
                  f"Avg Max distance: {np.mean(D_sorted[:, -1]):.3f} ± {np.std(D_sorted[:, -1]):.3f}\n")
        if isinstance(self.s, float):
            return np.exp(-sqD / (2 * self.s ** 2))
        else:
            return np.exp(-sqD / 2)

    def K_matrix_const(self, X1, X2=None, dim_pos=None, verbose=False):
        """Computes the kernel matrix of a Gaussian kernel + constant given two matrices of points"""
        if X2 is None:
            X2 = X1
        if isinstance(self.s, float) or isinstance(self.s, int):
            D = sqdist(X1, X2)
        elif isinstance(self.s, list) or isinstance(self.s, np.ndarray):
            D_pos = sqdist_weighted(X1[:, :dim_pos], X2[:, :dim_pos], self.s[:dim_pos])
            D_ori = sqdist_weighted(X1[:, dim_pos:], X2[:, dim_pos:], self.s[dim_pos:])

        if isinstance(self.s, float):
            return np.exp(-D / (2 * self.s ** 2)) + 1
        else:
            return (np.exp(-D_ori / 2) + 1) * ((np.exp(-D_pos / 2) + 1))

    def psi(self, x):
        """ Feature map """
        return np.hstack((np.atleast_2d(np.ones(x.shape[0])).T, x, np.sin(x), np.cos(x),
                          np.sin(x) * np.cos(x), np.sin(x) ** 2, np.cos(x) ** 2,
                          np.sin(x) ** 2 * np.cos(x), np.sin(x) * np.cos(x) ** 2, np.sin(x) ** 3, np.cos(x) ** 3))

    @classmethod
    def preprocess(cls, x):
        x[:, :3] = x[:, :3] * 1e2

    @classmethod
    def postprocess(cls, x):
        x[:, :3] = x[:, :3] / 1e2

def invPD(M):
    "Inverts positive definite matrix with Cholesky decomposition"
    C = np.linalg.inv(np.linalg.cholesky(M))
    return np.dot(C.T, C)
