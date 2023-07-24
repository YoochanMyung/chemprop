import os
import chemprop
import sys
import argparse
from typing import List, Set, Tuple, Union
import pandas as pd

def is_classification(csv_file):
    input_pd = pd.read_csv(csv_file)
    components = set(input_pd['label'].to_list())
    if len(components) > 5:
        return False
    else:
        return True

def run_Chemprop(args):
    type_of_run = str()

    label = args.label
    db_dir = args.db_dir
    add_feat_dir = args.add_feat_dir
    save_path = args.save_path
    use_rdkit = args.use_rdkit
    use_additional_feats = args.use_add_feats
    use_mcc = args.use_mcc
    use_train_only = args.use_train_only
    no_scaling = args.no_scaling
    layer_size = args.layer_size
    ensemble_size = args.ensemble_size

    train_path = os.path.join(db_dir,'{}_train.csv'.format(label))
    test_path =  os.path.join(db_dir,'{}_test.csv'.format(label)) 

    if use_additional_feats:
        train_feat_path = os.path.join(add_feat_dir, '{}_train.csv'.format(label)) 
        val_feat_path =  os.path.join(add_feat_dir, '{}_test.csv'.format(label)) 
        test_feat_path =  os.path.join(add_feat_dir, '{}_test.csv'.format(label)) 
        
        add_args = [
        '--features_path', train_feat_path,
        '--separate_val_features_path', test_feat_path,
        '--separate_test_features_path', test_feat_path
        ]
        #         add_args = [
        # '--features_generator', 'deeppk_bee_tox'
        # ]

    save_dir = os.path.join(save_path,label)

    if is_classification(test_path):
        type_of_run = 'classification'
    else:
        type_of_run = 'regression'

    arguments = [
        '--smiles_columns','smiles_standarized',
        '--target_columns','label',
        '--data_path', train_path,
        '--dataset_type', type_of_run,
        '--save_dir', save_dir,
        '--separate_val_path', test_path,
        '--separate_test_path', test_path,
        # '--ffn_num_layers', str(layer_size),
        # '--ensemble_size', str(ensemble_size),
        '--num_workers','8',
        '--num_folds','5',
        # '--epochs', '30',
        # '--batch_size', '128',
        '--quiet',
        '--show_individual_scores',
        '--save_smiles_splits',
        '--save_preds'
        ]

    extra_arg = [
        '--features_generator','rdkit_2d_normalized', # additional
        ]

    if type_of_run == 'classification':
        if use_mcc:
            arguments = arguments + ['--loss_function', 'mcc']
        else:
            arguments = arguments + ['--loss_function', 'binary_cross_entropy']
            
        arguments = arguments + ['--metric', 'mcc', '--extra_metrics', 'f1', 'auc', 'accuracy']
    else:
        arguments = arguments + ['--metric', 'rmse', '--extra_metrics', 'r2', 'mse', 'mae', '--loss_function', 'mse']

    if use_rdkit:
        arguments = arguments + extra_arg

    if use_additional_feats:
        arguments = arguments + add_args    

    if no_scaling:
        arguments = arguments + ['--no_features_scaling']

    try:
        args = chemprop.args.TrainArgs().parse_args(arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    except Exception as e:
        print(label, e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ex) python run_chemprop_wosplit.py pyriformis_reg ./smiles .')
    parser.add_argument("label", help="Select a label to train",type=str)
    parser.add_argument("db_dir", help="Provide a directory of CSVs",type=str)
    parser.add_argument("save_path", help="Choose dirpath",type=str)

    # Optional    
    parser.add_argument("-ensemble_size", help="Set the size of ensemble", default=1, type=str)
    parser.add_argument("-layer_size", help="Set the size of layer", default=2, type=str)
    parser.add_argument("--use_rdkit", help="Use rdkit feats", action='store_true')
    parser.add_argument("--use_add_feats", help="Use additional feats", action='store_true')
    parser.add_argument("--add_feat_dir", help="Provide a directory of CSVs",type=str)
    parser.add_argument("--use_mcc", help="Use mcc for loss function", action='store_true')
    parser.add_argument("--use_train_only", help="Use trainval for training", action='store_true')
    parser.add_argument("--no_scaling", help="Use mcc for loss function", action='store_true')

    args = parser.parse_args()
    run_Chemprop(args)
