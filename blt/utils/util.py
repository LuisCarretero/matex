import torch
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from io import open
import os
import os.path as osp
import pickle

from utils.networks import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def models_save(model, logpath):
    torch.save(model.state_dict(), logpath)


def models_load(model, loaddir):
    return model.load_state_dict(torch.load(loaddir, map_location=device))


def save_pkl(data, logpath):
    with open(logpath, 'wb') as f:
        pickle.dump(data, f)


def load_pkl(logpath):
    with open(logpath, 'rb') as input_file:
        pkl_data = pickle.load(input_file)
    return pkl_data


def data_save(demos, in_dist_goals, ood_goals, logpath):
    with open(osp.join(logpath, 'demos.pkl'), 'wb') as f:
        pickle.dump(demos, f)
    with open(osp.join(logpath, 'in_dist_goals.pkl'), 'wb') as f:
        pickle.dump(in_dist_goals, f)
    with open(osp.join(logpath, 'ood_goals.pkl'), 'wb') as f:
        pickle.dump(ood_goals, f)


def data_load(loaddir):
    with open(osp.join(loaddir, 'demos.pkl'), 'rb') as input_file:
        demos = pickle.load(input_file)
    with open(osp.join(loaddir, 'in_dist_goals.pkl'), 'rb') as input_file:
        in_dist_eval_goals = pickle.load(input_file)
    with open(osp.join(loaddir, 'ood_goals.pkl'), 'rb') as input_file:
        ood_eval_goals = pickle.load(input_file)
    return demos, in_dist_eval_goals, ood_eval_goals


def define_model(model_type, obs_size, ac_size, hidden_layer_size, feature_dim, hidden_depth):
    if model_type == 'mlp':
        model = MlpPredictor(obs_size, ac_size, hidden_layer_size, hidden_depth)
    elif model_type == 'bilinear':
        model = BilinearPredictor(obs_size, ac_size, hidden_layer_size, feature_dim, hidden_depth)
    elif model_type == 'bilinear_scalardelta':
        model = BilinearPredictorScalarDelta(obs_size, ac_size, hidden_layer_size, feature_dim, hidden_depth)
    else:
        print('model_type', model_type)
        raise NotImplementedError('not implemented other policies')
    return model


def eval_supervised(model_type, model, logdir, eval_dataset, similarity_type, transducer=None,
                    use_dom_know_eval=False, eval_type='', verbose=True, write_results=True):
    test_X, test_Y, test_formula = eval_dataset['test_X'], eval_dataset['test_Y'], eval_dataset['test_formula']
    errors = []
    preds = {'preds': [], 'gt': test_Y, 'anchor_idxs': [], 'train_analogy_pair_idx': []}
    tmp_preds = {'preds': [], 'gt': test_Y, 'anchor_idxs': [], 'train_analogy_pair_idx': []}
    for k in range(len(test_X)):
        if verbose:
            print(f'{eval_type} ', k+1, '/', len(test_X))
        curr_test = test_X[k][None]
        gt_output = test_Y[k]
        #get delta
        if model_type in ['nn', 'bilinear', 'bilinear_scalardelta']:
            closest_train, anchor_idx, train_analogy_pair_idx = transducer.choose_anchor(
                curr_test, test_formula[k], use_dom_know_eval, return_anchor=True, verbose=verbose)
            preds['anchor_idxs'].append(anchor_idx)
            preds['train_analogy_pair_idx'].append(train_analogy_pair_idx)
            tmp_preds['anchor_idxs'].append(anchor_idx)
            tmp_preds['train_analogy_pair_idx'].append(train_analogy_pair_idx)
            closest_train = closest_train[None]
            delta = get_deltas(closest_train, curr_test, similarity_type)
        #eval with model
        if model_type == 'mlp':
            y = model(torch.Tensor(curr_test).to(device)).cpu().detach().numpy()[0]
        elif model_type == 'bilinear' or model_type == 'bilinear_scalardelta':
            y = model(torch.Tensor(closest_train).to(device), torch.Tensor(delta).to(device)).cpu().detach().numpy()[0]

        preds['preds'].append(y)
        loss = np.linalg.norm(gt_output - y)
        errors.append(loss)
        if verbose:
            print('gt ', gt_output, ' pred ', y, ' loss ', np.round(loss,3))
        if write_results and k % 50 == 0:
            tmp_preds['preds'] = np.array(preds['preds'])
            save_pkl(preds, logpath=osp.join(logdir, model_type+f'_{eval_type}_tmp'+'.pkl'))

    errors = np.array(errors)
    mean = round(np.mean(errors),4)
    sem = round(np.std(errors, ddof=1) / np.sqrt(np.size(errors)),4)
    result_line = f'MAE: {mean} ± {sem}\n'
    if verbose:
        print(result_line)
    if write_results:
        results_path = os.path.join(logdir, 'results.txt')
        with open(results_path, 'a') as f:
            f.write(f"Evaluation Type: {eval_type}\n")
            f.write(result_line)
            f.write('\n')
    preds['preds'] = np.array(preds['preds'])
    preds['mae'] = mean
    preds['sem'] = sem
    return preds


def get_deltas(X1, X2, similarity_type):
    # X1, X2 [batch x feat]
    if similarity_type == 'subtraction':
        # direction is important
        return X2 - X1 
    elif similarity_type == 'cosine':
        #dot product each row. cosine in [0,2]. normalization.
        if X1.dtype == np.float64:
            return (1 - (np.sum(X1*X2, axis=1) / (np.linalg.norm(X1, axis=1)*np.linalg.norm(X2, axis=1)))).reshape(-1,1)
        return (1 - (torch.sum(X1*X2, axis=1) / (torch.linalg.norm(X1, axis=1)*torch.linalg.norm(X2, axis=1)))).reshape(-1,1)


def compute_gradients(parameters):
    total_gradient_norm = None
    for p in parameters:
        if p.grad is None:
             continue
        current = p.grad.data.norm(2) ** 2
        if total_gradient_norm is None:
            total_gradient_norm = current
        else:
            total_gradient_norm += current
    return total_gradient_norm ** 0.5


def compute_params(parameters):
    total_param_norm = None
    for p in parameters:
        current = p.data.norm(2) ** 2
        if total_param_norm is None:
            total_param_norm = current
        else:
            total_param_norm += current
    return total_param_norm ** 0.5


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text


def log_params(model):
    # save number of model parameters
    total_params = sum(param.numel() for param in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params_not_trainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    return total_params, params_trainable, params_not_trainable

