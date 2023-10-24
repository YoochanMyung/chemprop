import argparse
import os
import chemprop
import simplejson as json
from check_wandb import fetch_result
import pandas as pd
import sys
sys.path.append('/home/ymyung/projects/deeppk/src/chemprop')

def check_categorical(input_csv):
    input_pd = pd.read_csv(input_csv)
    components = set(input_pd['label'].to_list())
    if len(components) > 5:
        return False
    else:
        return True

def run(kwargs):
    csv_dir = '/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/5_using_other_DBs/dataset/admetlab2'

    if check_categorical(os.path.join('/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/5_using_other_DBs/dataset/admetlab2',f'{kwargs.endpoint}_test.csv')):
        best_run = fetch_result({'target': kwargs.endpoint, 'r': False, 'run_name': kwargs.run_name})
    else:
        best_run = fetch_result({'target': kwargs.endpoint, 'r': True, 'run_name': kwargs.run_name})
        
    print("Working on: {}".format(kwargs.endpoint))
    run_name = best_run['run_name']
    if kwargs.run_name:
        save_dir = os.path.join(kwargs.save_path, kwargs.endpoint, kwargs.run_name)
    else:  
        save_dir = os.path.join(kwargs.save_path, kwargs.endpoint, run_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    result_json = json.load(open(os.path.join(f'/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/5_using_other_DBs/after_hyperopt/admetlab2/{kwargs.endpoint}/{run_name}/{kwargs.endpoint}_best.json'), 'r'))
    result_json = {k: str(v) for k, v in result_json.items()}

    init_lr = float(result_json['max_lr']) * float(result_json['init_lr_ratio'])
    final_lr = float(result_json['max_lr']) * float(result_json['final_lr_ratio'])

    arguments = [
    '--smiles_columns','smiles_standarized',
    '--target_columns','label',
    '--dataset_type', 'classification',
    ## Input paths
    '--data_path', os.path.join(csv_dir, f'{kwargs.endpoint}_train.csv'),
    '--separate_val_path', os.path.join(csv_dir, f'{kwargs.endpoint}_val.csv'),
    '--separate_test_path', os.path.join(csv_dir, f'{kwargs.endpoint}_test.csv'),
    '--save_dir', save_dir,
    ## Model arguments
    '--depth', result_json['depth'],
    '--init_lr', str(init_lr),
    # '--init_lr_ratio', result_json['init_lr_ratio'],
    '--max_lr', result_json['max_lr'],
    '--final_lr', str(final_lr),
    '--dropout', result_json['dropout'],
    '--activation', result_json['activation'],
    '--batch_size', result_json['batch_size'],
    '--aggregation', result_json['aggregation'],
    '--hidden_size', result_json['hidden_size'],
    '--warmup_epochs', result_json["warmup_epochs"],
    '--ffn_num_layers', result_json['ffn_num_layers'],
    '--ffn_hidden_size', result_json['ffn_hidden_size'],
    '--aggregation_norm', result_json["aggregation_norm"],
    '--metric','mcc',
    '--num_folds','3',
    '--extra_metrics', 'f1','auc','accuracy',
    '--save_preds',
    '--num_workers','8']

    if kwargs.mordred or result_json['mordred'] == 'True':
        arguments = arguments + [
        '--features_path', os.path.join(csv_dir, 'mordred', f'{kwargs.endpoint}_train.csv'),  
        '--separate_val_features_path', os.path.join(csv_dir, 'mordred', f'{kwargs.endpoint}_val.csv'),  
        '--separate_test_features_path', os.path.join(csv_dir, 'mordred', f'{kwargs.endpoint}_test.csv'),  
        ]

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

    # print(arguments)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ex) python build_hypopt_model.py skin_sens . -run_name silvery-dust-1001')
    parser.add_argument("endpoint", help="Choose endpoint name",type=str)
    parser.add_argument("save_path", help="Choose dir path for save",type=str)
    parser.add_argument("-reg", action='store_true', help="")
    parser.add_argument("-mordred", action='store_true', help="Run type between regression and classification")
    parser.add_argument("-run_name",type=str)
    
    args = parser.parse_args()
    run(args)
