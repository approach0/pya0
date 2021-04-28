import numpy as np
from sklearn.tree import DecisionTreeRegressor
from numba import jit
import warnings
import pickle
warnings.filterwarnings('ignore')


class LambdaMART:

    def __init__(self, num_trees=100, max_depth=5, learning_rate=0.01):
        self.num_trees = num_trees
        self.max_depth = max_depth

        self.lr = learning_rate
        self.eps = 0.000001 # smooth factor

        self.trees = []
        self.gamma = np.zeros((num_trees, 2**(max_depth + 1)))

    @staticmethod
    def _calc_DCG(rel_arr):
        i = np.arange(1, len(rel_arr) + 1) # rank
        numerator = 2 ** rel_arr - 1
        denominator = 1 / np.log2(i + 1)
        return np.dot(numerator, denominator)

    @staticmethod
    def _calc_NDCG_at_K(F_q, rels_q, K):
        # calcualte model DCG
        model_ranks = np.argsort(F_q)[::-1]
        model_rels_q = rels_q[model_ranks][0:K]
        DCG = LambdaMART()._calc_DCG(model_rels_q)
        # calcualte model NDCG
        perfect_rels = np.sort(rels_q)[::-1][0:K]
        maxDCG = LambdaMART()._calc_DCG(perfect_rels)
        NDCG = DCG / maxDCG if maxDCG != 0 else 0
        return NDCG

    @jit
    def _calc_results(self, qid, rels, F):
        # qid: scaler ID of query ID
        # rels: labels of relevance for all results of this qid
        # F: base model scores for all results of this qid

        # perfect NDCG
        # TODO: it only depends on `rels', can be accelerated by cache.
        perfect_rels = np.sort(rels)[::-1] # perfect label order
        maxDCG = LambdaMART()._calc_DCG(perfect_rels)
        if maxDCG != 0:
            NDCG_wo_log = (2 ** rels - 1) / maxDCG
        else:
            NDCG_wo_log = np.zeros(len(rels))

        # model single DCG log part
        model_ranks = np.argsort(F)[::-1] + 1
        log_order = 1 / np.log2(1 + model_ranks)

        # absolute Delta NDCG
        abs_Delta = np.zeros((len(rels), len(rels)))

        # for every pair combinations ...
        for i in range(len(rels)):
            for j in range(i + 1, len(rels)):
                # only learn those with S_ij = 1
                if rels[i] != rels[j]:
                    log_swap = np.copy(log_order)
                    log_swap[i], log_swap[j] = log_swap[j], log_swap[i]
                    delta_score = np.dot(NDCG_wo_log, log_swap - log_order)
                    if rels[i] > rels[j]:
                        abs_Delta[i, j] = delta_score
                    else:
                        abs_Delta[j, i] = delta_score
        abs_Delta = np.abs(abs_Delta)

        # calculate first and second derivatives (Lambda and Omega)
        Oij = np.reshape(F, (-1, 1)) - np.reshape(F, (1, -1)) # o - o^T = O_ij
        Rho = 1 / (1 + np.exp(Oij))
        abs_Delta_Rho = abs_Delta * Rho
        Lambda = - abs_Delta_Rho
        Omega = abs_Delta_Rho * (1 - Rho)

        # expand to two terms according to Chain Rule ...
        Lambda -= Lambda.T # lambda_i = lambda_ij - lambda_ji
        Omega += Omega.T # omega_i = 2 * omega_i
        # (because -(d lambda)/(d o_j) is positive)

        # sum elements in each column
        lambda_i = np.sum(Lambda, axis=0)
        omega_i = np.sum(Omega, axis=0)

        return lambda_i, omega_i

    def fit(self, X, rels, qids):
        n_rows = np.shape(X)[0]
        F = np.zeros(n_rows) # base model i.e., F(x_i) = o_i
        for m in range(self.num_trees):
            print(f'building {m}-th tree...')
            Lambda = np.array([])
            Omega = np.array([])
            for q in np.unique(qids):
                rels_q = rels[q == qids]
                F_q = F[q == qids]
                Lambda_q, Omega_q = self._calc_results(q, rels_q, F_q)
                Lambda = np.append(Lambda, Lambda_q)
                Omega = np.append(Omega, Omega_q)

            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, Lambda)
            self.trees.append(tree)
            leaves = tree.apply(X) # get R_jm to which x_i maps
            for leaf in np.unique(leaves):
                # compute scalar gamma
                I = (leaves == leaf)
                gamma = np.sum(Lambda[I]) / (np.sum(Omega[I]) + self.eps)
                # save gamma
                self.gamma[m, leaf] = gamma
                # improve the model
                F += self.lr * I * gamma
            # evaluate current training NDCGs
            self.evaluate(X, rels, qids)

    def predict(self, X):
        n_rows = np.shape(X)[0]
        F = np.zeros(n_rows) # base model i.e., F(x_i) = o_i
        for m in range(len(self.trees)):
            leaves = self.trees[m].apply(X)
            for leaf in np.unique(leaves):
                I = (leaves == leaf)
                F += self.lr * I * self.gamma[m, leaf]
        return F

    def evaluate(self, X, rels, qids):
        F = self.predict(X)
        uniq_qids = np.unique(qids)
        N = len(uniq_qids)
        acc_NDCG1 = 0
        acc_NDCG10 = 0
        acc_NDCG100 = 0
        for q in uniq_qids:
            rels_q = rels[q == qids]
            F_q = F[q == qids]
            # accumulate the sum NDCG across queries
            acc_NDCG1 += self._calc_NDCG_at_K(F_q, rels_q, 1)
            acc_NDCG10 += self._calc_NDCG_at_K(F_q, rels_q, 10)
            acc_NDCG100 += self._calc_NDCG_at_K(F_q, rels_q, 100)
        print(f'avg NDCG@1: {acc_NDCG1 / N}')
        print(f'avg NDCG@10: {acc_NDCG10 / N}')
        print(f'avg NDCG@100: {acc_NDCG100 / N}')

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)


if __name__ == '__main__':
    import os
    from sklearn.datasets import load_svmlight_file
    # you can download MQ2008 dataset as a getting-started example
    train_data = load_svmlight_file('tmp.train.dat', query_id = True)
    test_data = load_svmlight_file('tmp.test.dat', query_id = True)
    train_features, train_rel, train_qid = train_data
    test_features, test_rel, test_qid = test_data

    if os.path.exists('./model.pkl'):
        # load model
        model = LambdaMART.load('./model.pkl')
        model.evaluate(test_features, test_rel, test_qid)
    else:
        # train model
        model = LambdaMART()
        model.fit(train_features, train_rel, train_qid)

        #print('=== test evaluation ===')
        #model.evaluate(test_features, test_rel, test_qid)
        model.save('./model.pkl')
