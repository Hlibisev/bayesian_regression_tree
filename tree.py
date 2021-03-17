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

    def define(self, split_dim, split_value, left, right, posterior):
        self.split_dim = split_dim
        self.split_value = split_value
        self.left = left
        self.right = right

    def is_leaf(self):
        return (self.left is None) and (self.right is None)


class BayesianDecisionTree:
    def __init__(self,  max_depth: int = 20, min_samples_leaf: int = 1,
                 max_iter: int = np.inf, feat_bag: bool = False, freq_feat_bag: float = None, partition_prior=0.9):
        """
        :param max_depth: max depth of tree
        :param min_samples_leaf: how many samples should be at leaf to avoid separation
        :param max_iter: max iterations at finding best split values
        :param feat_bag: use feature bagging?
        :param freq_feat_bag: what part of feature is used at separation. Basically sqrt(num_feat)
        """
        self.root = None
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth if max_depth is not None else np.inf
        self.max_iter = max_iter
        self.feat_bag = feat_bag
        self.freq_feat_bag = freq_feat_bag
        self.number_feat_bag = None
        self.partition_prior = partition_prior

    def fit(self, X, y):
        if self.freq_feat_bag is None:
            self.number_feat_bag = int(np.sqrt(X.shape[1]))
        else:
            self.number_feat_bag = int(X.shape[1] * self.freq_feat_bag)

        self.root = DecisionTreeNode(ind=np.arange(len(X)))
        viewed_nodes = {self.root}  # type: Set[DecisionTreeNode]

        while viewed_nodes:
            new_nodes = set()
            for node in viewed_nodes:
                if node.size > 2 * self.min_samples_leaf and node.depth < self.max_depth:
                    split_dim, split_value, posterior_left, posterior_right = self.find_split_and_posteriors(X[node.ind], y[node.ind], node)

                    # if partition was not happened , this node is leaf
                    if split_dim is None:
                        node.posterior = self.get_posterior(node, y[node.ind])
                        continue

                    sep = X[node.ind, split_dim] < split_value
                    left_ind, right_ind = node.ind[sep], node.ind[~sep]

                    left, right = self.make_nodes(node, split_dim, split_value, left_ind,
                                                  right_ind, posterior_left, posterior_right)

                    new_nodes.add(left)
                    new_nodes.add(right)
            viewed_nodes = new_nodes

    @staticmethod
    def make_nodes(node, split_dim, split_value, left_ind, right_ind, posterior_left, posterior_right):
        left = DecisionTreeNode(ind=left_ind, depth=node.depth + 1, prior=posterior_left)
        right = DecisionTreeNode(ind=right_ind, depth=node.depth + 1, prior=posterior_right)
        node.define(split_dim, split_value, left, right)
        return left, right

    def predict_proba(self, X):
        ind = np.arange(len(X))
        return self._traversal(X, ind, self.root)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return [max(p.keys(), key=lambda k: p[k]) for p in probabilities]

    def find_split_and_posteriors(self, X: np.ndarray, y: np.ndarray, node: DecisionTreeNode):
        best_log_data = self.get_log_p_data_before_split(node, y)
        split_dim, split_value, best_posterior_left, best_posterior_right = None, None, None, None

        n_features = X.shape[1]
        for feature in range(n_features):
            X_sorted_indexes = np.argsort(X[:, feature])
            X_f_sorted_by_feature = X[X_sorted_indexes, feature]
            y_sorted_by_feature = y[X_sorted_indexes]

            values_X, indexes_of_unique = np.unique(X_f_sorted_by_feature, return_index=True)

            # max iterations is equal max_iters
            if len(values_X) > self.max_iter:
                indexes = sorted(np.random.choice(len(values_X), self.max_iter, replace=False))
            else:
                indexes = indexes_of_unique

            n = len(y)
            n_splits = len(indexes)

            posterior_left, posterior_right = self.get_posteriors_for_split_node(node, y_sorted_by_feature, indexes)
            log_p_data_after_split = self.get_log_p_data_after_split(node, posterior_left, posterior_right, n_splits, n)
            index_max = log_p_data_after_split.argmax()

            if log_p_data_after_split[index_max] > best_log_data:
                # remember new best split
                best_log_data = log_p_data_after_split[index_max]
                split_value = y_sorted_by_feature[index_max - 1: index_max].mean()
                split_dim = feature
                best_posterior_left, best_posterior_right = posterior_left, posterior_right

        return split_dim, split_value, best_posterior_left, best_posterior_right

    def _traversal(self, X, ind, node):
        if node.is_leaf():
            values, counts = np.unique(self.y[node.ind], return_counts=True)
            freq = counts / sum(counts)
            predict = dict(zip(values, freq))
            return [predict for _ in range(len(ind))]

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
        log_p_prior = np.log(1 - self.partition_prior ** (1 + node.depth))
        log_p_data = self._get_log_p_data(node, alpha_post, beta_post, kappa_post, n)

        return log_p_prior + log_p_data

    def get_log_p_data_after_split(self, node: DecisionTreeNode, posterior_for_left_node, posterior_for_right_node, n_splits, n_dim):
        n = len(y)
        n1 = np.arange(1, n)
        n2 = n - n1

        mu1, kappa1, alpha1, beta1 = posterior_for_left_node
        mu2, kappa2, alpha2, beta2 = posterior_for_right_node

        log_p_prior = np.log(self.partition_prior ** (1 + node.depth) / (n_splits * n_dim))
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



X = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 2)
y = np.array([1, 2, 3, 4, 5])
print(X.shape, y.shape)
tree = BayesianDecisionTree()
tree.fit(X, y)

