import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances_argmin_min
from utils.util import get_deltas


class Transducer():
    def __init__(self, samples, **kwargs):
        self.samples = samples


class DeltaDistributionTransducer(Transducer):
    """ samples stored/computed training deltas and gets anchor point at inference"""
    def __init__(self, samples, train_deltas, skew, n_approx, sample_deltas, sample_train, type_idxs, similarity_type, verbose=True, **kwargs):
        super().__init__(samples=samples)

        # compute deltas once from training samples
        self.train_X = samples['train_X']
        self.train_Y = samples['train_Y']
        self.formula_X = samples['train_formula']
        self.type_idxs = type_idxs
        self.similarity_type = similarity_type
        self.sample_train = sample_train
        self.verbose = verbose

        # deltas that were actually used in training
        if sample_deltas:
            if verbose:
                print('sampling train deltas transducer')
            self.train_deltas = train_deltas[random.sample(range(train_deltas.shape[0]), train_deltas.shape[0]//10)] #sample from train deltas
        else:
            self.train_deltas = train_deltas

        # approx training  deltas
        if verbose:
            print('approximating deltas')
        if len(train_deltas) == 0: # not stored during training
            t1_idx = np.random.randint(len(self.train_X), size=(n_approx,)) # Indices of A
            t2_idx = np.random.randint(len(self.train_X), size=(n_approx,)) # Indices of sample B
            # priviledged training, knowledge of the Y distribution's skewness (transformation type/object class type)
            if skew == 'right': # t2 have higher ys than t1
                swap_idxs = (self.train_Y[t1_idx] > self.train_Y[t2_idx]).flatten()
                t1_idx[swap_idxs], t2_idx[swap_idxs] = t2_idx[swap_idxs], t1_idx[swap_idxs]
            elif skew == 'left':
                swap_idxs = (self.train_Y[t1_idx] < self.train_Y[t2_idx]).flatten()
                t1_idx[swap_idxs], t2_idx[swap_idxs] = t2_idx[swap_idxs], t1_idx[swap_idxs]
            self.train_deltas = get_deltas(self.train_X[t1_idx], self.train_X[t2_idx], similarity_type)
            self.train_pairs = np.stack([t1_idx, t2_idx], axis=1)
    
    
    def choose_anchor(self, curr_obs, curr_formula, use_dom_know_eval=False, return_anchor=False, exhaustive_search=True, eps_percentile=10, verbose=True):
        """return idx for training sample that gives delta closest to training deltas"""

        if use_dom_know_eval: # priviledged eval
            sample_idxs_of_type_obs = np.where(np.argwhere(self.train_X[:,self.type_idxs]) == np.argwhere(curr_obs[self.type_idxs])[0])[0]
        elif self.sample_train:
            sample_idxs_of_type_obs = np.random.randint(len(self.train_X), size=(len(self.train_X)//5,))
        else:
            sample_idxs_of_type_obs = list(range(len(self.train_X)))
        
        # closest delta in dist
        curr_deltas = get_deltas(self.train_X[sample_idxs_of_type_obs], curr_obs, self.similarity_type) #size(len(train_X), num_feat)
        if exhaustive_search:
            # Streaming equivalent of: np.unravel_index(argmin(cdist(curr_deltas, self.train_deltas)))
            # Avoids allocating the full (n_train, n_train_deltas) distance matrix.
            best_per_anchor, dist_per_anchor = pairwise_distances_argmin_min(
                curr_deltas, self.train_deltas, metric='euclidean'
            )
            anchor_idx = int(np.argmin(dist_per_anchor))
            delta_idx = int(best_per_anchor[anchor_idx])
            closest_obs = self.train_X[anchor_idx]
            train_analogy_pair_idx = self.train_pairs[delta_idx]
        else:
            delta_eps = np.percentile([np.min(np.linalg.norm(train_d - self.train_deltas, axis=1)) for train_d in curr_deltas[np.random.choice(len(curr_deltas), size=100)]], eps_percentile)
            while True:
                # Sampling a train delta
                anchor_idx = np.random.choice(len(curr_deltas))
                closest_obs = curr_deltas[anchor_idx]
                min_dist = np.min(np.linalg.norm(closest_obs - self.train_deltas, axis=1))
                if min_dist <= delta_eps:
                    break

        if verbose:
            print('test material ', curr_formula, ' anchor material ', self.formula_X[anchor_idx], '\n train pair analogy ', self.formula_X[train_analogy_pair_idx])
        
        if return_anchor:
            return closest_obs, anchor_idx, train_analogy_pair_idx
        return closest_obs


def define_transducer(samples, train_deltas, skew, n_approx, sample_deltas=False, sample_train=False, \
                      type_idxs=None, similarity_type=None, verbose=True):
    """return instance of transducer class"""
    transducer = DeltaDistributionTransducer(samples=samples, train_deltas=train_deltas, skew=skew, \
                                             n_approx=n_approx, sample_deltas=sample_deltas, \
                                             sample_train=sample_train, type_idxs=type_idxs,
                                             similarity_type=similarity_type, verbose=verbose)
    return transducer
