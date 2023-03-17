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

regression_tasks = list()
classification_tasks = list()

def check_categorical(input_pd):
    components = set(input_pd['label'].to_list())
    if len(components) > 5:
        return False
    else:
        return True

def parallelize_dataframe(p_list, func, num_cores=10):
    p_list_split = np.array_split(p_list, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, p_list_split)
    pool.close()
    pool.join()

def run_train(smi_list):
    for each_smi in smi_list:
        print("Working on: {}".format(each_smi))
        data_path = each_smi
        property_name = each_smi.split('/')[-1].split('.csv')[0]
        save_dir = os.path.join('/home/ymyung/projects/deeppk/3_Results/',property_name)
        arguments = [
        '--smiles_columns','smiles_standarized',
        '--data_path', data_path,
        '--target_columns','label',
        '--dataset_type', 'classification',
        '--loss_function','mcc',
        '--save_dir', save_dir,
        '--epochs', '20',
        # '--split_type','scaffold_balanced',
        '--split_type','random_with_repeated_smiles',
        '--num_folds','5',
        '--save_smiles_splits',
        '--quiet',
        '--show_individual_scores',
        '--metric','mcc',
        '--extra_metrics', 'f1','auc','accuracy',
        '--save_preds',
        # '--num_workers','2',
        #'--ensemble_size', '5',
        # '--gpu',0,
        '--features_generator','rdkit_2d_normalized', # additional
        '--no_features_scaling' # additional    
        ]

        try:
            args = chemprop.args.TrainArgs().parse_args(arguments)
            mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
        except Exception as e:
            print(each_smi, e)


def run_test(smi_list):
    for each_smi in smi_list:
        print("Working on: {}".format(each_smi))
        property_name = each_smi.split('/')[-1].split('.csv')[0]
        save_dir = os.path.join('/home/ymyung/projects/deeppk/3_Results/all_new',property_name)
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
            print(each_smi, e)


if __name__ == '__main__':
    for each_csv in all_smiles_list:
        if check_categorical(pd.read_csv(each_csv,sep=',')):
            classification_tasks.append(each_csv)
        else:
            regression_tasks.append(each_csv)
    # parallelize_dataframe(classification_tasks, run_train)
    #run_train(classification_tasks)
    run_test(classification_tasks)
