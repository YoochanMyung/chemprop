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

# regression_list = ['bp','clearance','cyppro_iinh','cyppro_xi_cyp1a2','cyppro_xi_cyp2a9',\

#regression_list = ['bp','clearance','cyppro_iinh','cyppro_xi_cyp1a2','cyppro_xi_cyp2a9',\
#        'cyppro_xi_cyp2c19','cyppro_xi_cyp2d6','human_clinical_drugs_vdss','hydration_free_energy',\
#        'log_d7.4','log_p','log_s','log_vp','mdck_permeability','mp','pka_acid','pka_basic','ppb']
regression_list = ['log_p']

dir_path = '/home/ymyung/projects/deeppk/1_dataset/compared/scaffold_split'

def parallelize_dataframe(p_list, func, num_cores=3):
    p_list_split = np.array_split(p_list, num_cores)
    pool = Pool(num_cores)
    df = pool.map(func, p_list_split)
    pool.close()
    pool.join()

def run_hyperopt(p_list):
    for each in p_list:
        save_dir = os.path.join('/home/ymyung/projects/deeppk/3_Results/regression_MPNN_results','6_hypopt',each)
        arguments = [
        '--data_path', os.path.join(dir_path,each,'fold_0','train_full.csv'),
        '--separate_val_path',os.path.join(dir_path,each,'fold_0','val_full.csv'),
        '--separate_test_path',os.path.join(dir_path,each,'fold_0','test_full.csv'),
        '--target_columns','label',
        '--dataset_type', 'regression',
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

run_hyperopt(regression_list)
#parallelize_dataframe(regression_list,run_hyperopt, 4)
