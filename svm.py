#Copy Right @Maxwell Jianzhou Wang
#ELG5131 Graphical Models Project 2
#Main Phase --- Support Vector Machine for both Trainer and Predictor

import numpy as np
import numpy.linalg as la
import cvxopt.solvers

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVM_Trainer(object):
    def __init__(self, k, c):
        self._k = k #kernel functions
        self._c = c

    def trainer(self, x, y):
        lagrange_multipliers = self._compute_multipliers(x, y)
        return self._construct_predictor(x, y, lagrange_multipliers)

    def _gram_matrix(self, x):
        n_samples, n_features = x.shape
        mat = np.zeros((n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                mat[i, j] = self._kernel(x_i, x_j)
        return mat

    def _construct_predictor(self, x, y, lagrange_multipliers):
        n_samples, n_features = x.shape
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_vector_labels = y[support_vector_indices]
        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]


        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._k,
                bias= 0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self._k,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _compute_multipliers(self, x, y):
        n_samples, n_features = X.shape

        mat = self._gram_matrix(x)

        P = cvxopt.matrix(np.outer(y, y) * mat)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])

class SVM_Predictor(object):
    def __init__(self, k, bias, weights, support_vectors, support_vector_signs):
        self.k = k
        self.bias = bias
        self.weights = weights
        self.support_vectors = support_vectors
        self.support_vectors_signs = support_vector_signs

    def predict (self, x):
        result = self._bias
        for x_i, y_i, z_i in zip( self.weight, self.support_vectors, self.support_vectors):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result)

class Kernel(object):
    def linear(self):
        return lambda x, y: np.dot(x,y)
    def rbf(gamma):
        return lambda x, y: np.exp(-gamma*la.norm(np.substract(x,y)))
















