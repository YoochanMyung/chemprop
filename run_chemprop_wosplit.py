import os
import chemprop
import sys
import argparse
from typing import List, Set, Tuple, Union

def check_categorical(input_pd):
    components = set(input_pd['label'].to_list())
    if len(components) > 5:
        return False
    else:
        return True

def def_split_sizes(train_ratio: float) -> List[str] :
    train_f = train_ratio
    val_f = (1- train_f)/2
    test_f = val_f

    return [train_f, val_f, test_f]

#         run_classification(label, db_dir, save_path, loss_fn, use_rdkit)

def run_classification(label, db_dir, save_path, loss_fn, use_rdkit):
    train_path = os.path.join(db_dir,'{}_train.csv'.format(label)) 
    val_path =  os.path.join(db_dir,'{}_val.csv'.format(label)) 
    test_path =  os.path.join(db_dir,'{}_test.csv'.format(label)) 

    if loss_fn == 'default' or loss_fn == 'bce':
        loss_fn = 'binary_cross_entropy'

    save_dir = os.path.join(save_path,label)
    arguments = [
    '--smiles_columns','smiles_standarized',
    '--data_path', train_path,
    '--target_columns','label',
    '--dataset_type', 'classification',
    '--loss_function',loss_fn,
    '--save_dir', save_dir,
    '--epochs', '50',
    '--separate_val_path', val_path,
    '--separate_test_path', test_path,
    '--num_folds','10',
    '--save_smiles_splits',
    '--quiet',
    '--show_individual_scores',
    '--metric','mcc',
    '--extra_metrics', 'f1','auc','accuracy',
    '--save_preds',
    '--num_workers','4']

    extra_arg = [
    '--features_generator','rdkit_2d_normalized', # additional
    '--no_features_scaling' # additional
    ]

    if use_rdkit:
        arguments = arguments + extra_arg

    try:
        args = chemprop.args.TrainArgs().parse_args(arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    except Exception as e:
        print(label, e)


# def run_regression(smi, save_path, split_type, train_ratio, loss_fn, use_rdki):
def run_regression(label, db_dir, save_path, loss_fn, use_rdkit):
    train_path = os.path.join(db_dir,'{}_train.csv'.format(label)) 
    val_path =  os.path.join(db_dir,'{}_val.csv'.format(label)) 
    test_path =  os.path.join(db_dir,'{}_test.csv'.format(label)) 

    if loss_fn == 'default':
        loss_fn = 'mse'

    save_dir = os.path.join(save_path, label)
    arguments = [
    '--smiles_columns','smiles_standarized',
    '--data_path', train_path,
    '--target_columns','label',
    '--dataset_type', 'regression',
    '--loss_function', loss_fn,
    '--save_dir', save_dir,
    '--epochs', '50',
    '--separate_val_path', val_path,
    '--separate_test_path', test_path,
    '--num_folds','10',
    '--save_smiles_splits',
    '--metric', 'r2', # or rmse
    '--extra_metrics', 'rmse', 'mse','mae',
    '--quiet',
    '--num_workers','4',
    '--show_individual_scores',
    '--save_preds']

    extra_args = [
    '--features_generator','rdkit_2d_normalized', # additional
    '--no_features_scaling' # additional,
    ]

    if use_rdkit:
        arguments = arguments + extra_args

    try:
        args = chemprop.args.TrainArgs().parse_args(arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    except Exception as e:
        print(label, e)


# def run_test(smi,save_path,use_rdkit):
def run_test(label, db_dir, save_path, use_rdkit):
    # print("Working on: {}".format(smi))
    # property_name = smi.split('/')[-1].split('.csv')[0]
    save_dir = os.path.join(save_path, label)

    arguments = [
    '--test_path', '{}/fold_4/test_smiles.csv'.format(save_dir),
    '--preds_path', '{}/test_preds_cla.csv'.format(save_dir),
    '--checkpoint_dir', save_dir]

    extra_args = [
    '--features_generator','rdkit_2d_normalized', # additional
    '--no_features_scaling' # additional
    ]

    if use_rdkit:
        arguments = arguments + extra_args

    try:
        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args)
    except Exception as e:
        print(label, e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ex) python run_chemprop.py aaa.csv ./classification -run_type classification -split_type Random -loss_fn mcc -use_rdkit')
    parser.add_argument("label", help="Select a label to train",type=str)
    parser.add_argument("db_dir", help="Provide a directory of CSVs",type=str)
    parser.add_argument("save_path", help="Choose dirpath",type=str)
    parser.add_argument("-run_type", help="Run type between regression and classification",choices=['classification','regression'], default="classification")
    # parser.add_argument("-split_type", help="Random or Scaffold",choices=['random','scaffold'], default="random")
    # parser.add_argument("-train_ratio", help="Training ratio", default=0.8, type=float)
    parser.add_argument("-loss_fn", help="[bce,mcc,mse] default is binary cross entropy for classification and mse for regression ",choices=['bce','mcc','mse'], default="default")
    parser.add_argument("-use_rdkit", help="Choose the number of shuffling", action='store_true')

    args = parser.parse_args()

    label = args.label
    db_dir = args.db_dir
    save_path = args.save_path
    run_type = args.run_type
    loss_fn = args.loss_fn
    use_rdkit = args.use_rdkit

    if run_type == 'regression':
        run_regression(label, db_dir, save_path, loss_fn, use_rdkit)
        # run_test(label, db_dir, save_path, use_rdkit)
    else:
        run_classification(label, db_dir, save_path, loss_fn, use_rdkit)
        # run_test(label, db_dir, save_path, use_rdkit)




# run_classification('ames', '/home/ymyung/projects/deeppk/1_dataset/pk_data/processed_data/ANALYSIS/classification/train_val_test/random_split', '/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/test2', 'bce', False)
# run_regression('bp', '/home/ymyung/projects/deeppk/1_dataset/pk_data/processed_data/ANALYSIS/regression/train_val_test/random_split', '/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/test2', 'mse', False)