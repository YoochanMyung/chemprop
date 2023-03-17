import os
import chemprop
import pandas as pd
import glob
import numpy as np
from multiprocessing import Pool

#dir_path = '/home/ymyung/projects/deeppk/1_dataset/pk_data/processed_data/Alex/all/datasets'
# dir_path = '/home/ymyung/projects/deeppk/1_dataset/compared/scaffold_split'
# dir_path = '/home/ymyung/projects/deeppk/1_dataset/pk_data/processed_data/all_new/6_deeppk_tox'
dir_path = '/home/ymyung/projects/deeppk/2_ML/Datasets'
# all_smiles_list = glob.glob(os.path.join(dir_path,'**/*.smi'))
# all_smiles_list = [smi for smi in all_smiles_list if 'Murckos' in smi]
# all_smiles_list = [smi for smi in all_smiles_list if 'interpretable' in smi]
# all_smiles_list = [smi for smi in all_smiles_list if not 'toxicity' in smi]
# all_smiles_list= glob.glob(os.path.join(dir_path,'**/fold_0/*.csv'),recursive=True)
all_smiles_list= glob.glob(os.path.join(dir_path,'*.csv'),recursive=True)

def check_categorical(input_pd):
    components = set(input_pd['label'].to_list())
    if len(components) > 5:
        return False
    else:
        return True   


regression_tasks = list()
classification_tasks = list()

for each_csv in all_smiles_list:
    # print(each_csv)
    if check_categorical(pd.read_csv(each_csv,sep=',')):
        classification_tasks.append(each_csv)
    else:
        regression_tasks.append(each_csv)


def parallelize_dataframe(p_list, func, num_cores=10):
    p_list_split = np.array_split(p_list, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, p_list_split)
    pool.close()
    pool.join()

def run_train(smi_list):
    for each_smi in smi_list:
        property_name = each_smi.split('/')[-1].split('.csv')[0]
        save_dir = os.path.join('/home/ymyung/projects/deeppk/3_Results/all_new/',property_name)
        arguments = [
        '--smiles_columns','smiles_standarized',
        '--data_path', each_smi,
        '--target_columns','label',
        '--dataset_type', 'regression',
        # '--loss_function','mse',
        '--save_dir', save_dir,
        '--epochs', '20',
        # '--split_type','scaffold_balanced',
        '--split_type','random_with_repeated_smiles',
        '--num_folds','5',
        '--save_smiles_splits',
        '--metric', 'rmse',
        '--extra_metrics', 'r2', 'mse','mae',
        '--quiet',
        # '--num_workers','2',
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
            print(each_smi, e)

        arguments = [
        '--test_path', '{}/fold_0/test_smiles.csv'.format(save_dir),
        '--preds_path', '{}/test_preds_reg.csv'.format(save_dir),
        '--checkpoint_dir', save_dir,
        '--num_workers','2',
        '--features_generator','rdkit_2d_normalized', # additional
        '--no_features_scaling' # additional
        ]

def run_test(smi_list):
    for each_smi in smi_list:
        print(each_smi)
        property_name = each_smi.split('/')[-1].split('.csv')[0]
        save_dir = os.path.join('/home/ymyung/projects/deeppk/3_Results/all_new/',property_name)
        
        arguments = [
        '--test_path', '{}/fold_4/test_smiles.csv'.format(save_dir),
        '--preds_path', '{}/test_preds_reg.csv'.format(save_dir),
        '--checkpoint_dir', save_dir,
        '--features_generator','rdkit_2d_normalized', # additional
        '--no_features_scaling' # additional
        ]

        try:
            args = chemprop.args.PredictArgs().parse_args(arguments)
            preds = chemprop.train.make_predictions(args=args)
        except Exception as e:
            print(each_smi, e)    

if __name__ == '__main__':
    for each_csv in all_smiles_list:
        if check_categorical(pd.read_csv(each_csv,sep=',')):
            classification_tasks.append(each_csv)
        else:
            regression_tasks.append(each_csv)

    # parallelize_dataframe(regression_tasks, run_train)
    # run_train(regression_tasks)
    run_test(regression_tasks)