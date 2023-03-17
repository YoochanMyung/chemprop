import os
import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA
import scipy 
import glob

from multiprocessing import Pool
import numpy as np

#classification_list = ['oct2','bbb','bcrp','caco_2','cyp1a2_inhibitor','cyp2c19_inhibitor','cyp2c9_inhibitor','cyp2c9_substrate','cyp2d6_inhibitor','cyp2d6_substrate','cyp3a4_inhibitor','cyp3a4_substrate','f20','f30','hia','oatp1b1','oatp1b3','ob','pgp_inhibitor','pgp_substrate','skin_permeability','t0.5']

classification_list = ['cyp1a2_inhibitor','cyp2d6_inhibitor']
dir_path = '/home/ymyung/projects/deeppk/1_dataset/compared/scaffold_split'

def parallelize_dataframe(p_list, func, num_cores=3):
    p_list_split = np.array_split(p_list, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, p_list_split)
    pool.close()
    pool.join()

def run_hyperopt(p_list):
    for each in p_list:
        save_dir = os.path.join('/home/ymyung/projects/deeppk/3_Results/classification_MPNN_results','6_hypopt',each)
        arguments = [
        '--data_path', os.path.join(dir_path,each,'fold_0','train_full.csv'),
        '--separate_val_path',os.path.join(dir_path,each,'fold_0','val_full.csv'),
        '--separate_test_path',os.path.join(dir_path,each,'fold_0','test_full.csv'),
        '--target_columns','label',
        '--dataset_type', 'classification',
        '--num_iters', '10',
        '--log_dir',save_dir,
        '--config_save_path',os.path.join(save_dir,'config.json'),
        '--features_generator','rdkit_2d_normalized', # additional
        '--no_features_scaling', # additional,
        '--show_individual_scores',
        '--quiet'    
        ]
        from chemprop.hyperparameter_optimization import chemprop_hyperopt

        args = chemprop.args.HyperoptArgs().parse_args(arguments)
        chemprop.hyperparameter_optimization.hyperopt(args = args)


#parallelize_dataframe(classification_list,run_hyperopt, 6)
run_hyperopt(classification_list)
