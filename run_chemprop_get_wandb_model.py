import os
import chemprop
import sys
import argparse
from typing import List, Set, Tuple, Union
import pandas as pd
import json

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
    save_path = args.save_path

    use_rdkit = False
    use_additional_feats = True
    use_mcc = False
    
    wandb_json = args.wandb_json
    wandb_config = json.load(open(wandb_json, 'rb'))

    selected_bias = wandb_config['bias']['value']
    selected_depth = wandb_config['depth']['value']
    selected_dropout = wandb_config['dropout']['value']
    selected_activation = wandb_config['activation']['value']
    selected_batch_size = wandb_config['batch_size']['value']
    selected_aggregation = wandb_config['aggregation']['value']
    selected_hidden_size = wandb_config['hidden_size']['value']
    selected_ffn_num_layers = wandb_config['ffn_num_layers']['value']
    selected_weights_ffn_num_layers = wandb_config['weights_ffn_num_layers']['value']

    # use_rdkit = args.use_rdkit
    # use_additional_feats = args.use_add_feats
    # use_mcc = args.use_mcc
    # layer_size = args.layer_size
    # ensemble_size = args.ensemble_size

    train_path = os.path.join(db_dir,'{}_train.csv'.format(label)) 
    val_path =  os.path.join(db_dir,'{}_val.csv'.format(label)) 
    test_path =  os.path.join(db_dir,'{}_test.csv'.format(label)) 

    if is_classification(test_path):
        type_of_run = 'classification'
    else:
        type_of_run = 'regression'

    train_feat_path = os.path.join(f'/home/ymyung/projects/deeppk/2_ML_running/1_Greedy/Feature_engineering/data/6_final2/wo_smiles/{type_of_run}/powertransform/non_redundant_features/train_val_test/no_ID_label', '{}_train.csv'.format(label)) 
    val_feat_path =  os.path.join(f'/home/ymyung/projects/deeppk/2_ML_running/1_Greedy/Feature_engineering/data/6_final2/wo_smiles/{type_of_run}/powertransform/non_redundant_features/train_val_test/no_ID_label', '{}_val.csv'.format(label)) 
    test_feat_path =  os.path.join(f'/home/ymyung/projects/deeppk/2_ML_running/1_Greedy/Feature_engineering/data/6_final2/wo_smiles/{type_of_run}/powertransform/non_redundant_features/train_val_test/no_ID_label', '{}_test.csv'.format(label)) 

    # train_feat_path = os.path.join(f'/home/ymyung/projects/deeppk/2_ML_running/1_Greedy/Feature_engineering/data/6_final2/wo_smiles/{type_of_run}/train_val_test/no_ID_label', '{}_train.csv'.format(label)) 
    # val_feat_path =  os.path.join(f'/home/ymyung/projects/deeppk/2_ML_running/1_Greedy/Feature_engineering/data/6_final2/wo_smiles/{type_of_run}/train_val_test/no_ID_label', '{}_val.csv'.format(label)) 
    # test_feat_path =  os.path.join(f'/home/ymyung/projects/deeppk/2_ML_running/1_Greedy/Feature_engineering/data/6_final2/wo_smiles/{type_of_run}/train_val_test/no_ID_label', '{}_test.csv'.format(label)) 

    save_dir = os.path.join(save_path,label)
    arguments = [
        '--smiles_columns','smiles_standarized',
        '--target_columns','label',
        '--data_path', train_path,
        '--dataset_type', type_of_run,
        '--save_dir', save_dir,
        '--separate_val_path', val_path,
        '--separate_test_path', test_path,
        '--num_workers','8',
        '--quiet',
        '--show_individual_scores',
        '--save_smiles_splits',
        '--save_preds',
        '--num_folds',str(4)
        ]

    add_args = [
        '--max_lr', str(0.1),
        '--no_features_scaling',
        '--features_path', train_feat_path,
        '--separate_val_features_path', val_feat_path,
        '--separate_test_features_path', test_feat_path,
        '--activation', str(selected_activation), # sweep_config
        '--aggregation', str(selected_aggregation), # sweep_config
        '--weights_ffn_num_layers', str(selected_weights_ffn_num_layers), # sweep_config
        '--depth', str(selected_depth), #sweep_config
        '--hidden_size', str(selected_hidden_size), #sweep_config
        '--dropout', str(selected_dropout), #sweep_config
        '--ffn_num_layers', str(selected_ffn_num_layers), #sweep_config
        '--epochs', str(50),
        '--batch_size', str(selected_batch_size), #sweep_config
        ]

    if type_of_run == 'classification':
        if use_mcc:
            arguments = arguments + ['--loss_function', 'mcc']
        else:
            arguments = arguments + ['--loss_function', 'binary_cross_entropy']
            
        arguments = arguments + ['--metric', 'mcc', '--extra_metrics', 'f1', 'auc', 'accuracy']
    else:
        arguments = arguments + ['--metric', 'rmse', '--extra_metrics', 'r2', 'mse', 'mae', '--loss_function', 'mse']

    if use_additional_feats:
        arguments = arguments + add_args    

    if selected_bias:
        arguments = arguments + ['--bias']

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
    parser.add_argument("wandb_json", help="Choose json file",type=str)

    # Optional    
    # parser.add_argument("-ensemble_size", help="Set the size of ensemble", default=1, type=str)
    # parser.add_argument("-layer_size", help="Set the size of layer", default=2, type=str)
    # parser.add_argument("--use_rdkit", help="Use rdkit feats", action='store_true')
    # parser.add_argument("--use_add_feats", help="Use additional feats", action='store_true')
    # parser.add_argument("--use_mcc", help="Use mcc for loss function", action='store_true')

    args = parser.parse_args()

    run_Chemprop(args)