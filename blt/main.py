import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import os.path as osp
import warnings
warnings.filterwarnings('ignore')
from io import open
import random
from ruamel.yaml import YAML
import json
import argparse
import datetime
# import wandb

from utils.util import models_save, models_load, save_pkl, load_pkl, define_model, eval_supervised
from utils.trainer import train_supervised
from utils.transducers import define_transducer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_supervised_training_and_eval(args, logdir='log'):
    """load data and model, train and evaluate"""

    yaml = YAML()
    v = yaml.load(open(osp.join('configs', 'materials.yml')))

    # Environment
    seed = v['env']['seed'] if args.seed is None else args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Model
    num_epochs = v['model']['num_epochs']
    use_dom_know_train = v['model']['use_dom_know_train']
    use_dom_know_eval = v['model']['use_dom_know_eval']
    store_train_deltas = v['model']['store_train_deltas']
    sample_deltas = v['model']['sample_deltas']
    sample_train = v['model']['sample_train']
    skew = v['model']['skew']
    mul_approx_train_deltas = v['model']['mul_approx_train_deltas']
    if not use_dom_know_train: skew = None

    date_string = datetime.datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    base_logdir = osp.join(logdir, args.dataset_name, args.prop_type, f'{args.data_filename}_{args.similarity_type}_{args.model_type}_hsize{str(args.hidden_layer_size)}_hnum{str(args.hidden_depth)}_esize{str(args.embedding_dim)}_bsize{str(args.batch_size)}')
    
    if args.model_path is None:
        logdir = osp.join(base_logdir, date_string)
        os.makedirs(logdir, exist_ok=True)
        print('logdir', logdir) 
        #if file exists will add to it and not overwrite it
        if not os.path.isfile(osp.join(logdir, 'config.txt')):
            with open(osp.join(logdir, 'config.txt'), 'a') as f:
                json.dump(v, f, indent=2)
    
        # #wandb offline logging
        # WANDB_DIR = osp.join(logdir, 'wandb')
        # os.mkdir(WANDB_DIR)
        # os.environ['WANDB_MODE'] = 'offline'
        # wandb.login()
        # run = wandb.init(
        #     dir=WANDB_DIR,
        #     # Set the project where this run will be logged
        #     project="Matex",
        #     # Name the run
        #     name=f'{args.dataset_name}_{ args.prop_type}_{args.data_filename}',
        #     # Track hyperparameters and run metadata
        #     config={
        #         'data_filename': args.data_filename,
        #         'similarity_type': args.similarity_type,
        #         'model_type': args.model_type,
        #         'hidden_size': args.hidden_layer_size,
        #         'hidden_num': args.hidden_depth,
        #         'embedding_size': args.embedding_dim,
        #         'batch_size': args.batch_size,
        #     })
    else: 
        logdir = osp.join(base_logdir, args.model_path)

    #Env
    obs_idxs = v['env']['obs_idxs']
    type_idxs = v['env']['type_idxs']
    y_size = v['env']['output_size']

    # Data
    print('loading data')
    samples = load_pkl(osp.join('data', args.dataset_name, args.prop_type, f'{args.data_filename}.pkl'))
    if obs_idxs is None: 
        obs_idxs = range(samples['train_X'].shape[-1])
        x_size = len(obs_idxs)
    assert (args.similarity_type == 'cosine' and args.model_type == 'bilinear_scalardelta') or args.similarity_type != 'cosine'
    n_approx_train_deltas = mul_approx_train_deltas * (len(samples['train_X'])) 
    print('number of deltas for approx:', n_approx_train_deltas)
    # add n_approx_train_deltas to config file:
    with open(osp.join(logdir, 'config.txt'), 'r') as f:
        config = json.load(f)
    config['n_approx_train_deltas'] = n_approx_train_deltas
    with open(osp.join(logdir, 'config.txt'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Model
    """
    model_type f(x)=y
    x=[s,g], dx=x-x'
    - bc on x
    - bilinear on x,dx
    """
    predictor = define_model(args.model_type, x_size, y_size, args.hidden_layer_size, args.embedding_dim, args.hidden_depth)
    print(predictor)
    predictor_path = osp.join(logdir, args.model_type+'.pt')
    train_deltas_path = osp.join(logdir, args.model_type+'_train_deltas'+'.pkl')
    transducer_deltas_path = osp.join(logdir, args.model_type+'_transducer_train_deltas'+'.pkl')
    #train and save
    if args.model_path is None:
        predictor, train_deltas = train_supervised(args.model_type, samples, predictor, logdir, obs_idxs, skew, num_epochs,
                                                    args.batch_size, checkpoint_path=logdir, store_train_deltas=store_train_deltas, 
                                                    similarity_type=args.similarity_type)
        models_save(predictor, logpath=predictor_path) # save learned models in logdir for later evaluation
        save_pkl(train_deltas, logpath=train_deltas_path) # save train_deltas for further evaluation.
        # transducer might approximate/sample train deltas                                                        
        # define transducer this is only used at test time (train is all to all)
        test_transducer = define_transducer(samples, train_deltas, skew, n_approx_train_deltas, sample_deltas=sample_deltas, \
                                            sample_train=sample_train, type_idxs=type_idxs, similarity_type=args.similarity_type)
        #save approx deltas used in eval
        save_pkl(test_transducer.train_deltas, logpath=transducer_deltas_path)
    else: #load model
        print('load model')
        models_load(predictor, predictor_path)
        predictor.to(device)
        train_deltas = []
        test_transducer = define_transducer(samples, train_deltas, skew, n_approx_train_deltas, sample_deltas=sample_deltas, \
                                                type_idxs=type_idxs, similarity_type=args.similarity_type)
        #save approx deltas used in eval
        print('saving aprrox deltas')
        save_pkl(test_transducer.train_deltas, logpath=transducer_deltas_path)
        
        #load deltas
        train_deltas = load_pkl(train_deltas_path)
        test_transducer = define_transducer(samples, train_deltas, skew, n_approx_train_deltas, sample_deltas=sample_deltas, 
                                            sample_train=sample_train, type_idxs=type_idxs, similarity_type=args.similarity_type)
        if sample_deltas:
            save_pkl(test_transducer.train_deltas, logpath=transducer_deltas_path)
    
    # Eval
    print('Eval!')
    predictor.eval()
    #eval in dist       
    plt.figure()
    eval_samples = eval_supervised(args.model_type, predictor, logdir, \
                            {'test_X': samples['eval_X'], 'test_Y': samples['eval_Y'], 'test_formula': samples['eval_formula']}, \
                            args.similarity_type, transducer=test_transducer, use_dom_know_eval=use_dom_know_eval, eval_type='val')
    save_pkl(eval_samples, logpath=osp.join(logdir, args.model_type+'_eval_in_dist'+'.pkl'))
    #eval ood
    plt.figure()
    eval_samples_ood = eval_supervised(args.model_type, predictor, logdir, \
                            {'test_X': samples['ood_X'], 'test_Y': samples['ood_Y'], 'test_formula': samples['ood_formula']}, \
                            args.similarity_type, transducer=test_transducer, use_dom_know_eval=use_dom_know_eval, eval_type='ood')
    save_pkl(eval_samples_ood, logpath=osp.join(logdir, args.model_type+'_eval_ood'+'.pkl'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='bilinear')
    parser.add_argument('--dataset_name', default='aflow')
    parser.add_argument('--prop_type', default='bulk_modulus_vrh')
    parser.add_argument('--similarity_type', default='subtraction')
    parser.add_argument('--data_filename', default='')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--hidden_layer_size', type=int, default=512)
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_depth', type=int, default=3)
    parser.add_argument('--debug', default=False)
    parser.add_argument('--model_path', default=None) #datetime
    parser.add_argument('--seed', type=int, default=None, help="Override seed in materials.yml")
    args = parser.parse_args()

    run_supervised_training_and_eval(args)
    