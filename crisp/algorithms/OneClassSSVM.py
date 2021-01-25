import numpy as np
import sys
from sklearn.base import BaseEstimator
from cvxopt import matrix as cvxopt_matrix
from cvxopt import solvers as cvxopt_solvers
from sklearn.base import BaseEstimator
import time
from pathlib import Path
from scipy.optimize import minimize
import pickle
from crisp.utils.planar_utils import sqdist
from sklearn.utils.validation import check_array, check_is_fitted



class OneClassSSVM(BaseEstimator):
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, s, v, balls=True, boundaries=None):
        self.balls = balls
        self.is_fitted_ = False
        self.k = 12
        self.c = None
        self.X = None
        self.Y = None
        self.s = s
        self.v = v
        self.boundaries = boundaries
        self.outdim = len(boundaries)

    def fit(self, X, y):
        """Computes best w using cutting plane algorithm
        Solves the following minimization problem using quadratic solvers
        from cvxopt:
        K \in R^{m \times m}
        a \in R^m
        1_v = [ 1, 1, ..., 1] \in R^m

        min_{c} 1/2 <a,Ka>
        s.t. 0 <= a <= 1/(v*m)
             <1_v,a> = 1

        ----------
        X : {array-like, sparse matrix}, shape (m, d_in)
            The training input samples.
        y : array-like, shape (m, d_out)
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
        m = X.shape[0]

        # CVXOPT solves problem in the form:
        #   min_c 1\2 <c,Pc> + <q, c>
        #   s.t. Gc <= h
        #        Ac =  b
        # Prepare the problem in such form
        P = cvxopt_matrix(self.K_matrix(X,y) / 2)   # <c,Kc>

        if self.balls:
            q = cvxopt_matrix(np.ones(m))
        else:
            q = cvxopt_matrix(np.zeros(m))           # <q,c>

        G = cvxopt_matrix(np.vstack((-np.identity(m), np.identity(m))))
        h = cvxopt_matrix(np.hstack((np.zeros(m), np.ones(m) / (self.v))))       # s.t. Gc <= h

        A = cvxopt_matrix(np.ones((1, m)))
        b = cvxopt_matrix(np.ones(1))

        # cvxopt parameters
        cvxopt_solvers.options['abstol'] = 1e-5
        cvxopt_solvers.options['reltol'] = 1e-5
        cvxopt_solvers.options['feastol'] = 1e-5

        if m > 1:
            print(f"Training OC-SVM model with {m} training points")
            t0 = time.time()
        sol = cvxopt_solvers.qp(P, q, G, h, A, b)
        self.c = np.array(sol['x'])
        if m > 1:
            t1 = time.time()
            print(f"Training completed in {(t1-t0)/60} mins")

        # `fit` should always return `self`
        return self

    def predict(self, X, y0=None, is_sequence=False,
                out=None):
        """ A reference implementation of a predicting function.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The test input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Returns an array of joints angles.
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X, accept_sparse=True)
        if out is None:
            out = print

        # y0 = np.array([(high-low)*np.random.random() + low for low, high in self.boundaries])

        y_hat = np.full((X.shape[0], self.outdim), np.inf)
        if X.shape[0] > 1:
            out(f"Predicting {X.shape[0]} test points")
            t0 = time.time()
        for idx, x in enumerate(X):
            if y0 is None and (not is_sequence or idx == 0):
                y0 = np.array([(high-low)*np.random.random() + low for low, high in self.boundaries])
            elif is_sequence and idx > 0:
                y0 = y_hat[idx-1]

            y_hat[idx] = minimize(self.f, x0=y0, args=(x[np.newaxis, :]), method='L-BFGS-B', bounds=self.boundaries, #jac=self.grad_f,
                                  options={'gtol': 1e-9, 'disp':False, 'maxiter':40000, 'maxfun':30000, 'ftol':1e-9, 'maxls':30}).x[np.newaxis, :]
            if not self.inside_boundaries(y_hat[idx]):
                print(f"[{idx}] outside of boundaries")
        if X.shape[0] > 1:
            t1 = time.time()
            out(f"Inference completed in {(t1-t0)/60} mins")

        return y_hat, {}

    def score(self, X_test, Y_test):
        """ Evaluates the squared loss over some test set """
        prediction = self.predict(X_test)  # predict
        return np.mean(np.sum((Y_test - prediction) ** 2, axis=1))

    def save(self, output_folder, save_name):
        """
        Saves the model in a .pickle file
        :param output_folder: destination folder
        :param save_name: name of the file
        """
        OC_SVM = {'c': self.c, 'X': self.X, 'Y':self.Y, 's': self.s, 'k':self.k, 'psi':self.psi, 'v':self.v}

        if save_name is None:
            save_name = f'OC_SVM.pickle'
        pickle.dump(OC_SVM, open(output_folder / save_name, 'wb'))

    def loss(self, y, y_hat):
        return np.sum(y**2 - y_hat**2)

    def psi(self, x, y):
        """ Structured feature map """
        return np.hstack((x, np.sin(x), np.cos(x), np.sin(x)*np.cos(x),
                          y, np.sin(y), np.cos(y), np.sin(y)*np.cos(y)))

    def inside_boundaries(self, Q):
        for q, bounds in zip(Q, self.boundaries):
            if q < bounds[0] or q > bounds[1]:
                return False
            return True

    def f(self, x, actual_x):
        """ Functional to be minimized for One Class Structured SVM.
        Reference:
         Bócsi, Botond, et al. "Learning tracking control with forward models."
         2012 IEEE International Conference on Robotics and Automation. IEEE, 2012."""
        if x.ndim == 1:
            x = x[np.newaxis, :]
        psi_tr = self.psi(self.X, self.Y)
        psi = self.psi(actual_x, x)
        D = sqdist(psi, psi_tr)
        return -np.dot(np.squeeze(self.c), np.exp(- D / (2*self.s**2)))

    def K_matrix(self, X, Y):
        """ Computes the kernel matrix wit the structured feature map """
        psi = self.psi(X, Y)
        D = sqdist(psi, psi)
        return np.exp(- D / (2*self.s**2))

    def grad_f(self, x, actual_x):
        """ Gradient of f w.r.t to x, needs testing """
        if x.ndim == 1:
            x = x[np.newaxis, :]
        psi_tr = self.psi(self.X, self.Y)
        psi = self.psi(actual_x, x)
        D = sqdist(psi, psi_tr)
        f_evals = (1/self.s**2) * np.exp(- D / (2*self.s**2))
        grads = x - self.Y - np.sin(self.Y) * np.cos(x) + np.cos(self.Y) * np.sin(x)
        return -np.dot(self.c.squeeze(), grads * f_evals[:, np.newaxis])

    def gd(self, y0, x):
        """ Gradient descent base implementation """
        y = y0
        for idx in range(500):
            y -= 1e+1*self.grad_f(y, x)
            if idx % 100 == 0:
                print(f"Grad norm at iter {idx}:\t {np.linalg.norm(self.grad_f(y, x))}")
        return y

    def load_state(self, path):
        """ Loads a model saved with 'save' function """
        try:
            state = pickle.load(open(path, 'rb'))
        except FileNotFoundError:
            print(f"OneClass_SVM.load_state(): Could not find the model to load, please check the path")
            sys.exit()
        self.s = state['s']
        self.c = state['c']
        self.X = state['X']
        self.Y = state['Y']
        self.k = state['k']
        self.v = state['v']
        self.psi = state['psi']
        self.is_fitted_ = True
