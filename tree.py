import numpy as np
from scipy.special import gammaln
from typing import Set


class DecisionTreeNode:
    def __init__(self, depth=0, ind=None, prior=[1, 1, 1, 1]):
        self.split_dim = None
        self.split_value = None
        self.left = None
        self.right = None
        self.ind = ind if ind is not None else []
        self.size = len(self.ind)
        self.depth = depth
        self.prior = prior
        self.posterior = None

    def define(self, split_dim, split_value, left, right):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


class BayesianDecisionTree:
    def __init__(self,  max_depth: int = 20, min_samples_leaf: int = 1, partition_prior=(0.9, 30), prior=(10, 10, 10, 100)):
        """
        :param max_depth: max depth of tree
        :param min_samples_leaf: how many samples should be at leaf to avoid separation
        """
        self.root = None
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.number_feat_bag = None
        self.partition_prior = partition_prior
        self.prior = prior

    def fit(self, X, y):
        self.root = DecisionTreeNode(ind=np.arange(len(X)), prior=self.prior)
        viewed_nodes = {self.root}  # type: Set[DecisionTreeNode]

        while viewed_nodes:
            new_nodes = set()
            for node in viewed_nodes:
                if node.size > 2 * self.min_samples_leaf and node.depth < self.max_depth:
                    split_dim, split_value = self.find_split(X[node.ind], y[node.ind], node)

                    # if partition was not happened , this node is leaf
                    if split_dim is None:
                        node.posterior = self.get_posterior(node, y[node.ind])
                        continue

                    sep = X[node.ind, split_dim] < split_value
                    left_ind, right_ind = node.ind[sep], node.ind[~sep]

                    if len(left_ind) == 1 or len(right_ind) == 1:
                        node.posterior = self.get_posterior(node, y[node.ind])
                        continue

                    posterior_left = self.get_posterior(node, y[left_ind])
                    posterior_right = self.get_posterior(node, y[right_ind])
                    # posterior_left = tuple([y[left_ind].mean()]) + posterior_left[1:]
                    # posterior_right = tuple([y[right_ind].mean()]) + posterior_right[1:]

                    # left, right = self.make_nodes(node, split_dim, split_value, left_ind,
                                                  # right_ind, node.prior, node.prior)

                    prior_left = self.set_prior(y, node)
                    prior_right = self.set_prior(y, node)

                    left, right = self.make_nodes(node, split_dim, split_value, left_ind,
                                                  right_ind, prior_left, prior_right)

                    new_nodes.add(left)
                    new_nodes.add(right)
                else:
                    node.posterior = self.get_posterior(node, y[node.ind])
            viewed_nodes = new_nodes

    @staticmethod
    def make_nodes(node, split_dim, split_value, left_ind, right_ind, posterior_left, posterior_right):
        left = DecisionTreeNode(ind=left_ind, depth=node.depth + 1, prior=posterior_left)
        right = DecisionTreeNode(ind=right_ind, depth=node.depth + 1, prior=posterior_right)
        node.define(split_dim, split_value, left, right)
        return left, right

    def predict(self, X):
        ind = np.arange(len(X))
        return self._traversal(X, ind, self.root)

    def find_split(self, X: np.ndarray, y: np.ndarray, node: DecisionTreeNode):
        best_log_data = self.get_log_p_data_before_split(node, y)
        split_dim, split_value, best_posterior_left, best_posterior_right = None, None, None, None

        n_features = X.shape[1]
        for feature in range(n_features):
            X_sorted_indexes = np.argsort(X[:, feature])
            X_f_sorted_by_feature = X[X_sorted_indexes, feature]
            y_sorted_by_feature = y[X_sorted_indexes]

            indexes_of_unique = np.unique(X_f_sorted_by_feature, return_index=True)[1][1:]

            posterior_left, posterior_right = self.get_posteriors_for_split_node(node, y_sorted_by_feature, indexes_of_unique)
            log_p_data_after_split = self.get_log_p_data_after_split(node, posterior_left, posterior_right, len(X_sorted_indexes))
            index_max = log_p_data_after_split.argmax()

            if log_p_data_after_split[index_max] > best_log_data:

                # remember new best split
                best_log_data = log_p_data_after_split[index_max]
                # split_value = X_f_sorted_by_feature[indexes_of_unique[index_max] - 1: indexes_of_unique[index_max]].mean()
                split_value = X_f_sorted_by_feature[indexes_of_unique[index_max: index_max + 1]].mean()
                split_dim = feature

        return split_dim, split_value

    @staticmethod
    def _mean_predict(node):
        return node.posterior[0]

    def _traversal(self, X, ind, node):
        if node.is_leaf():
            mean_predict = self._mean_predict(node)
            return np.zeros_like(ind) + mean_predict

        separation = X[ind, node.split_dim] < node.split_value
        left_ind, right_ind = ind[separation], ind[~separation]

        predict = np.zeros(len(ind), dtype=object)
        predict[separation] = self._traversal(X, left_ind, node.left)
        predict[~separation] = self._traversal(X, right_ind, node.right)
        return predict

    def _get_posterior(self, node, n, mean_y, variance_y):
        mu, kappa, alpha, beta = node.prior

        # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, equations (86) - (89)
        kappa_post = kappa + n
        mu_post = (kappa * mu + n * mean_y) / kappa_post
        alpha_post = alpha + 0.5 * n
        beta_post = beta + 0.5 * variance_y + 0.5 * kappa * n * (mean_y - mu) ** 2 / (kappa + n)

        return mu_post, kappa_post, alpha_post, beta_post

    def get_posterior(self, node, y):
        n = len(y)
        mean_y, variance_y_mul_n = self._comute_mean_and_var(y)
        return self._get_posterior(node, n, mean_y, variance_y_mul_n)

    def _get_log_p_data(self, node, alpha_new, beta_new, kappa_new, n_new):
        mu, kappa, alpha, beta = node.prior

        # see https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf, equation (95)
        return (gammaln(alpha_new) - gammaln(alpha)
                + alpha * np.log(beta) - alpha_new * np.log(beta_new)
                + 0.5 * np.log(kappa / kappa_new)
                - 0.5 * n_new * np.log(2 * np.pi))

    def get_log_p_data_before_split(self, node: DecisionTreeNode, y: np.ndarray):
        n = len(y)
        mean_y, variance_y_mul_n = self._comute_mean_and_var(y)
        mu_post, kappa_post, alpha_post, beta_post = self._get_posterior(node, n, mean_y, variance_y_mul_n)
        log_p_prior = self.get_log_p_prior(node.depth, n)
        log_p_data = self._get_log_p_data(node, alpha_post, beta_post, kappa_post, n)

        return log_p_prior + log_p_data

    def get_log_p_prior(self, depth, n1, n2=None):
        depth_penalty = self.partition_prior[0] ** (1 + depth)
        samples_penalty1 = 2 * np.arctan(n1 / self.partition_prior[1]) / np.pi
        if n2 is not None:
            samples_penalty2 = 2 * np.arctan(n2 / self.partition_prior[1]) / np.pi
            return np.log(depth_penalty * samples_penalty1 * samples_penalty2)
        return np.log(depth_penalty * samples_penalty1)

    def get_log_p_data_after_split(self, node: DecisionTreeNode, posterior_for_left_node, posterior_for_right_node, n):
        n1 = np.arange(1, n)
        n2 = n - n1

        mu1, kappa1, alpha1, beta1 = posterior_for_left_node
        mu2, kappa2, alpha2, beta2 = posterior_for_right_node

        log_p_prior = self.get_log_p_prior(node.depth + 1, n1, n2)
        log_p_data1 = self._get_log_p_data(node, alpha1, beta1, kappa1, n1)
        log_p_data2 = self._get_log_p_data(node, alpha2, beta2, kappa2, n2)

        return log_p_prior + log_p_data1 + log_p_data2

    def get_posteriors_for_split_node(self, node, y, split_indices):
        n = len(y)

        n1 = np.arange(1, n)
        n2 = n - n1
        sum1 = y.cumsum()[:-1]
        mean1 = sum1 / n1
        mean2 = (y.sum() - sum1) / n2
        y_minus_mean_sq_sum1 = ((y[:-1] - mean1) ** 2).cumsum()
        y_minus_mean_sq_sum2 = ((y[1:] - mean2)[::-1] ** 2).cumsum()[::-1]

        if len(split_indices) != len(y) - 1:
            # we are *not* splitting between all data points -> indexing necessary
            split_indices_minus_1 = split_indices - 1

            n1 = n1[split_indices_minus_1]
            n2 = n2[split_indices_minus_1]
            mean1 = mean1[split_indices_minus_1]
            mean2 = mean2[split_indices_minus_1]
            y_minus_mean_sq_sum1 = y_minus_mean_sq_sum1[split_indices_minus_1]
            y_minus_mean_sq_sum2 = y_minus_mean_sq_sum2[split_indices_minus_1]

        posterior_for_left_node = self._get_posterior(node, n1, mean1, y_minus_mean_sq_sum1)
        posterior_for_right_node = self._get_posterior(node, n2, mean2, y_minus_mean_sq_sum2)

        return posterior_for_left_node, posterior_for_right_node

    @staticmethod
    def _comute_mean_and_var(y):
        mean_y = y.mean()
        variance_y_mul_n = ((y - mean_y) ** 2).sum()
        return mean_y, variance_y_mul_n

    def set_prior(self, y, node):
        mu = (y.mean() + node.prior[0])/2
        sd_prior = y.std() / 10
        prior_obs = 1
        kappa = prior_obs
        alpha = prior_obs / 2
        var_prior = sd_prior ** 2
        tau_prior = 1 / var_prior
        beta = alpha / tau_prior
        return mu, kappa, alpha, beta

from sklearn.metrics import mean_absolute_error as mae

if __name__ == "__main__":
    # [ 2.33333333 23.          2.33333333 23.          0.95      ]
    n_data = 300

    X = np.linspace(-1, 3, n_data).reshape(-1, 1)
    y = X[:, 0] ** 2 - X[:, 0] + np.random.rand(n_data) + np.cos(X[:, 0])

    mu = y.mean()
    sd_prior = y.std()
    prior_obs = 2
    kappa = prior_obs
    alpha = prior_obs / 2
    var_prior = sd_prior ** 2
    tau_prior = 1 / var_prior
    beta = alpha / tau_prior
    partition = (0.9, 10)

    prior = [mu, kappa, alpha, beta]

    tree = BayesianDecisionTree()
    tree.fit(X, y)
    print(mae(y, tree.predict(X)))
    # print(tree.predict(X))
