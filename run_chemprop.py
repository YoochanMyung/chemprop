import os
import chemprop
import sys

def check_categorical(input_pd):
    components = set(input_pd['label'].to_list())
    if len(components) > 5:
        return False
    else:
        return True

def run_classification(smi, save_path, split_type='RANDOM'):
    print("Working on: {}".format(smi))
    if split_type.upper() == 'RANDOM':
        split_type = 'random_with_repeated_smiles'
    else:
        split_type = 'scaffold_balanced'

    data_path = smi
    property_name = smi.split('/')[-1].split('.csv')[0]
    save_dir = os.path.join(save_path,property_name)
    arguments = [
    '--smiles_columns','smiles_standarized',
    '--data_path', data_path,
    '--target_columns','label',
    '--dataset_type', 'classification',
    # '--loss_function','mcc',
    '--save_dir', save_dir,
    '--epochs', '50',
    '--split_type',split_type,
    '--num_folds','5',
    '--save_smiles_splits',
    '--quiet',
    '--show_individual_scores',
    '--metric','mcc',
    '--extra_metrics', 'f1','auc','accuracy',
    '--save_preds',
    '--num_workers','4',
    #'--ensemble_size', '5',
    # '--gpu',0,
    '--features_generator','rdkit_2d_normalized', # additional
    '--no_features_scaling' # additional
    ]

    try:
        args = chemprop.args.TrainArgs().parse_args(arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    except Exception as e:
        print(smi, e)


def run_regression(smi, save_path, split_type='RANDOM'):
    property_name = smi.split('/')[-1].split('.csv')[0]
    if split_type.upper() == 'RANDOM':
        split_type = 'random_with_repeated_smiles'
    else:
        split_type = 'scaffold_balanced'
    
    save_dir = os.path.join(save_path,property_name)
    arguments = [
    '--smiles_columns','smiles_standarized',
    '--data_path', smi,
    '--target_columns','label',
    '--dataset_type', 'regression',
    # '--loss_function','mse',
    '--save_dir', save_dir,
    '--epochs', '50',
    '--split_type',split_type,
    '--num_folds','5',
    '--save_smiles_splits',
    '--metric', 'rmse',
    '--extra_metrics', 'r2', 'mse','mae',
    '--quiet',
    '--num_workers','4',
    '--show_individual_scores',
    '--save_preds',
    # '--gpu',0,
    # '--ensemble_size', '5',
    '--features_generator','rdkit_2d_normalized', # additional
    '--no_features_scaling' # additional,
    ]
    try:
        args = chemprop.args.TrainArgs().parse_args(arguments)
        mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
    except Exception as e:
        print(smi, e)


def run_test(smi,save_path):
    print("Working on: {}".format(smi))
    property_name = smi.split('/')[-1].split('.csv')[0]
    save_dir = os.path.join(save_path,property_name)

    arguments = [
    '--test_path', '{}/fold_4/test_smiles.csv'.format(save_dir),
    '--preds_path', '{}/test_preds_cla.csv'.format(save_dir),
    '--checkpoint_dir', save_dir,
    '--features_generator','rdkit_2d_normalized', # additional
    '--no_features_scaling' # additional
    ]

    try:
        args = chemprop.args.PredictArgs().parse_args(arguments)
        preds = chemprop.train.make_predictions(args=args)
    except Exception as e:
        print(smi, e)


if __name__ == '__main__':
    input_csv = sys.argv[1]
    save_path = sys.argv[2]
    run_type = sys.argv[3]
    split_type = sys.argv[4]

    if run_type == 'regression':
        run_regression(input_csv,save_path,split_type)
        run_test(input_csv,save_path)
    else:
        run_classification(input_csv,save_path,split_type)
        run_test(input_csv,save_path)
