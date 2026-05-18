import os
# Environment variables for controlling the number of threads
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import sys
import numpy as np
import pandas as pd
import argparse
import warnings
import resource
import pickle
from pymatgen.core import Structure
from sklearn.model_selection import train_test_split

# imports for data processing
from data_utils import *

MATEX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(MATEX_DIR)
DATA_DIR = os.path.join(MATEX_DIR, 'blt/data')

import deepchem as dc
from deepchem.feat.molecule_featurizers import RDKitDescriptors

# Set resource limit for file descriptors
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

# Suppress warnings
warnings.simplefilter("ignore")
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Function to extract composition from a container
def extract_composition(container):
    return container.composition

# Function to get composition from structure dictionary
def get_comp_from_struc(structure_dict):
    structure = Structure.from_dict(structure_dict)
    return structure.composition

def modnet_preprocess(path, ood_ratio=0.05):
    """Main function to preprocess data for MODNet."""
    df_og = load_data(path, args.property)
    df_og = preprocess_composition(df_og, args.property)

    train_moddata, eval_moddata, ood_moddata = create_moddata(df_og, args.property, ood_ratio)
    train_set, eval_set, ood_set = get_featurized_sets(train_moddata, eval_moddata, ood_moddata)

    print(f'Train: {len(train_set)}, Eval: {len(eval_set)}, OOD: {len(ood_set)}')
    print(f'Target values in Train: {train_set[args.property].min()} - {train_set[args.property].max()}')
    print(f'Target values in OOD: {ood_set[args.property].min()} - {ood_set[args.property].max()}')

    handle_nan_values(train_set, eval_set, ood_set, args.nan_strategy, args.dataset_name)
    if args.property == 'band_gap': 
        handle_zero_targets(train_set, eval_set, ood_set, args.property)
    train, eval, ood = extract_and_save_composition(train_moddata, eval_moddata, ood_moddata, train_set, eval_set, ood_set, path)

    return normalize_data(train, eval, ood, args.property, args.scaler)


def molnet_preprocess(args):
    featurizer = RDKitDescriptors(is_normalized=True)
    if args.property == 'delaney':
        _, datasets, _ = dc.molnet.load_delaney(featurizer=featurizer) # featurizer only works for ecfp, graphconv, onehot
    elif args.property == 'freesolv':
        _, datasets, _ = dc.molnet.load_freesolv(featurizer=featurizer)
    elif args.property == 'lipo':
        _, datasets, _ = dc.molnet.load_lipo(featurizer=featurizer)
    elif args.property == 'bace':
        _, datasets, _ = dc.molnet.load_bace_regression(featurizer=featurizer)

    train, valid, test = datasets
    all_dataset = dc.data.NumpyDataset(
    np.concatenate([train.X, valid.X, test.X]),
    np.concatenate([train.y, valid.y, test.y]),
    np.concatenate([train.w, valid.w, test.w]),
    np.concatenate([train.ids, valid.ids, test.ids])
    )
    sorted_indices = np.argsort(all_dataset.y.ravel())
    n_samples = len(all_dataset)
    
    # split 5% OOD, 5% eval
    ood_size = int(0.05 * n_samples)
    ood_indices = sorted_indices[-ood_size:]

    remaining_indices = sorted_indices[:-ood_size]

    eval_size = int(0.05 * len(remaining_indices))
    eval_indices, train_indices = train_test_split(remaining_indices, test_size=(1 - eval_size / len(remaining_indices)))

    # Split the dataset into train, eval, and OOD
    train_dataset = dc.data.NumpyDataset(
        all_dataset.X[train_indices],
        all_dataset.y[train_indices],
        all_dataset.w[train_indices],
        all_dataset.ids[train_indices]
    )

    eval_dataset = dc.data.NumpyDataset(
        all_dataset.X[eval_indices],
        all_dataset.y[eval_indices],
        all_dataset.w[eval_indices],
        all_dataset.ids[eval_indices]
    )

    ood_dataset = dc.data.NumpyDataset(
        all_dataset.X[ood_indices],
        all_dataset.y[ood_indices],
        all_dataset.w[ood_indices],
        all_dataset.ids[ood_indices]
    )

    # save csv
    train_set  = save_dataset_as_csv(train_dataset, f'train_featurized.csv', args.dataset_name, args.property)
    eval_set = save_dataset_as_csv(eval_dataset, f'eval_featurized.csv', args.dataset_name, args.property)
    ood_set = save_dataset_as_csv(ood_dataset, f'ood_featurized.csv', args.dataset_name, args.property)

    train_set , eval_set, ood_set = handle_nan_values(train_set, eval_set, ood_set, args.nan_strategy, args.dataset_name)

    return train_set, eval_set, ood_set

def main(args, ood_ratio=0.05):
    """
    Main function to preprocess the data according to the command line arguments

    Parameters
    ----------
    args: Namespace
        Command line arguments
    ood_ratio: float
        Ratio of out-of-distribution samples to sample from the dataset, by default 0.05 as in paper

    Returns
    -------
    saves pkl of featurized data
    """
    path = os.path.join(DATA_DIR, args.dataset_name, args.property)
    os.makedirs(path, exist_ok=True)


    if args.dataset_name == 'matbench':
        train, eval, ood = modnet_preprocess(path, ood_ratio=ood_ratio)
        dataframes = {
            'train': train,
            'eval': eval,
            'ood': ood
        }
    elif args.dataset_name == 'molnet':
        train, eval, ood = molnet_preprocess(args)
        dataframes = {
            'train': train,
            'eval': eval,
            'ood': ood
        }

    dataset = {}
    for split_name in ['train', 'eval', 'ood']:
        if args.dataset_name == 'mp' or args.dataset_name == 'aflow':
            filename_feat_data = os.path.join(DATA_DIR, args.dataset_name, args.property, f'{split_name}.csv')
            data = pd.read_csv(filename_feat_data)
            X, y, formulae, _ = generate_features(data, elem_prop='oliynyk') 
        elif args.dataset_name == 'matbench':
            # filename_feat_data = os.path.join(config.DATA_DIR, args.dataset_name, args.property, f'{split_name}_featurized.csv')
            # data = pd.read_csv(filename_feat_data)
            split_df = dataframes[split_name]
            X, y, formulae = split_df.drop(columns=[f'{args.property}', 'composition']) , split_df[f'{args.property}'], split_df['composition']
        elif args.dataset_name == 'molnet':
            split_df = dataframes[split_name]
            X, y, formulae = np.vstack(split_df['rep'].to_numpy()) , split_df['target'], split_df['formula']

        if isinstance(X, np.ndarray):
            X = X
        else:
            X = X.to_numpy()

        dataset[split_name+'_X'] = X
        dataset[split_name+'_Y'] = y.to_numpy().reshape(-1,1)
        dataset[split_name+'_formula'] = formulae.to_numpy()

    # Build the filename
    filename_parts = []
    if args.dataset_name == 'matbench':
        filename_parts.append('magpie')
    elif args.dataset_name == 'mp' or args.dataset_name == 'aflow':
        filename_parts.append('oliynyk')
    elif args.dataset_name == 'molnet':
        filename_parts.append('rdkit')
    filename = '_'.join(filename_parts) + '.pkl'
    # save data
    save_path = os.path.join(DATA_DIR, args.dataset_name, args.property, filename)
    with open(save_path, 'wb') as f: 
        pickle.dump(dataset, f)

    print('data pkl splits saved!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='aflow', choices=['aflow', 'matbench', 'mp', 'molnet'])
    parser.add_argument('--property', default='bulk_modulus_vrh', choices=['bulk_modulus_vrh', 'debye_temperature', 'Egap', 'shear_modulus_vrh', 'thermal_conductivity_300K', 'thermal_expansion_300K', \
                                              'band_gap', 'refractive_ind', 'yield_strength', \
                                              'bulk_modulus', 'elastic_anisotropy', 'shear_modulus', \
                                              'bace', 'delaney', 'freesolv', 'lipo'])
    parser.add_argument('--nan_strategy', default='drop_sample', choices=['const', 'drop_feat', 'drop_sample'])
    parser.add_argument('--scaler', default='minmax', choices=['minmax', 'standard'])
    
    args = parser.parse_args()

    main(args)
    