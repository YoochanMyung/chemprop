import os
import pandas as pd
from typing import Any
import sys

import torch
from torch.optim.lr_scheduler import ExponentialLR

import argparse
import wandb
import socket

GLOBAL_ARGS = dict()
reg_label_list = ["bbb_cns", "bioconcF", "bp", "caco2", "caco2_logPaap", "cl", "fdamdd_reg", \
                  "fm_reg", "fu", "hydrationE", "lc50", "lc50dm", "ld50", "logbcf", "logd", \
                  "logp", "logs", "logvp", "mdck", "mp", "pka", "pkb", "ppb", "pyriformis_reg",\
                  "rat_acute_reg", "rat_chronic", "skin_permeability", "vd"]

def is_classification(csv_file):
    input_pd = pd.read_csv(csv_file)
    components = set(input_pd['label'].to_list())
    if len(components) > 5:
        return False
    else:
        return True
    
def run_Chemprop():
    if not GLOBAL_ARGS['no_wandb']:
        wandb.init()
    
    label = GLOBAL_ARGS['label']
    type_of_run = ''

    if label in reg_label_list:
        type_of_run = 'regression'
    else:
        type_of_run = 'classification'

    hostname = socket.gethostname()
    if hostname == 'ymyung-Precision-5820-Tower': # local
        sys.path.append('/home/ymyung/projects/deeppk/src/chemprop')
        smiles_dir = f'/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/test/admetlab2/'
    #     add_feats_dir = f'/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/test/admetlab2/full_features_only/'
        save_dir = f'/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/test/hyperopt'
    elif hostname == 'bio21hpc.bio21.unimelb.edu.au': # bio21_hpc
        sys.path.append('/home/ymyung/deeppk/src/chemprop')
        smiles_dir = f'/home/ymyung/deeppk/1_data/1_original/{type_of_run}/train_val_test/random_split'
        # add_feats_dir = f'/home/ymyung/deeppk/1_data/1_original/{type_of_run}/train_val_test/random_split/full_features_only'
        save_dir = f'/home/ymyung/deeppk/2_ML/sweeps'
    elif hostname == 'wiener.hpc.dc.uq.edu.au': # wiener
        sys.path.append('/clusterdata/uqymyung/src/chemprop')
        smiles_dir = f'/clusterdata/uqymyung/uqymyung/projects/deeppk/1_dataset/1_original/{type_of_run}/train_val_test/random_split'
        # add_feats_dir = f'/clusterdata/uqymyung/uqymyung/projects/deeppk/1_dataset/1_original/{type_of_run}/train_val_test/random_split/full_features_only'
        save_dir = f'/clusterdata/uqymyung/uqymyung/projects/deeppk/2_ML/1_Chemprop/1_MPNN/1_Random/sweeps'
    elif hostname == 'ymyung-Precision-Tower-5810': # local bio21
        sys.path.append('/home/ymyung/Projects/deeppk/src/chemprop')
        smiles_dir = '/home/ymyung/Projects/deeppk/data/toxcsm/'
        save_dir = '/home/ymyung/Projects/deeppk/runs/hyperopt_toxcsm'
    else:
        AssertionError('Wrong Platform')

    import chemprop
    from chemprop.train import train
    from chemprop.train.loss_functions import get_loss_func
    from chemprop.train.evaluate import evaluate, evaluate_predictions
    from chemprop.train.predict import predict
    from chemprop.constants import  TRAIN_LOGGER_NAME
    from chemprop.models import MoleculeModel
    from chemprop.data import get_class_sizes, get_data, MoleculeDataLoader, validate_dataset_type, get_task_names
    from chemprop.utils import build_optimizer, build_lr_scheduler, create_logger
    from chemprop.features import set_extra_atom_fdim, set_extra_bond_fdim

    train_path = os.path.join(smiles_dir,'{}_train.csv'.format(label)) 
    val_path =  os.path.join(smiles_dir,'{}_test.csv'.format(label)) 
    test_path =  os.path.join(smiles_dir,'{}_test.csv'.format(label)) 

    train_pt = os.path.join(smiles_dir,'{}_train_{}.pt'.format(label, wandb.config.add_feats)) 
    val_pt =  os.path.join(smiles_dir,'{}_val_{}.pt'.format(label, wandb.config.add_feats)) 
    test_pt =  os.path.join(smiles_dir,'{}_test_{}.pt'.format(label, wandb.config.add_feats))

    save_dir = os.path.join(save_dir,label)

    arguments = [
        '--smiles_columns','smiles_standarized',
        '--target_columns','label',
        '--num_workers','4',
        '--quiet',
        '--max_lr', str(0.1),
        '--data_path', train_path,
        '--dataset_type', type_of_run,
        '--save_dir', save_dir,
        '--separate_val_path', test_path,
        '--separate_test_path', test_path
        ]
    
    if GLOBAL_ARGS['no_wandb']:
        hyper_arguments = [
            '--activation', str('ReLU'), # sweep_config['parameters']
            '--aggregation', str('norm'), # sweep_config['parameters']
            '--weights_ffn_num_layers', str(3), # sweep_config['parameters']
            '--depth', str(5), #sweep_config['parameters']
            '--hidden_size', str(128), #sweep_config['parameters']
            '--dropout', str(0.5), #sweep_config['parameters']
            '--ffn_num_layers', str(4), #sweep_config['parameters']
            '--epochs', str(50), #sweep_config['parameters']
            '--batch_size', str(64), #sweep_config['parameters']
            ]
    else:
        hyper_arguments = [
            '--activation', str(wandb.config.activation), # sweep_config
            '--aggregation', str(wandb.config.aggregation), # sweep_config
            '--weights_ffn_num_layers', str(wandb.config.weights_ffn_num_layers), # sweep_config
            '--depth', str(wandb.config.depth), #sweep_config
            '--hidden_size', str(wandb.config.hidden_size), #sweep_config
            '--dropout', str(wandb.config.dropout), #sweep_config
            '--ffn_num_layers', str(wandb.config.ffn_num_layers), #sweep_config
            '--epochs', str(50),
            '--batch_size', str(wandb.config.batch_size), #sweep_config
            ]

    arguments = arguments + hyper_arguments

    if not GLOBAL_ARGS['no_wandb']:    
        if wandb.config.bias:
            arguments = arguments + ['--bias']

    if str(wandb.config.add_feats) == 'no':
        pass
    else:
        add_feats_dir = os.path.join(smiles_dir, str(wandb.config.add_feats))
        train_feat_path = os.path.join(add_feats_dir, '{}_train.csv'.format(label)) 
        val_feat_path =  os.path.join(add_feats_dir, '{}_test.csv'.format(label)) 
        test_feat_path =  os.path.join(add_feats_dir, '{}_test.csv'.format(label)) 

        add_feats = [
            '--features_path', train_feat_path,
            '--separate_val_features_path', test_feat_path,
            '--separate_test_features_path', test_feat_path
            ]

        arguments = arguments + add_feats

    if type_of_run == 'classification':
        arguments = arguments + ['--metric', 'mcc', '--extra_metrics', 'f1', 'auc', 'accuracy']
    else:
        arguments = arguments + ['--metric', 'rmse', '--extra_metrics', 'r2', 'mse', 'mae', '--loss_function', 'mse']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = chemprop.args.TrainArgs().parse_args(arguments)
    args.task_names = get_task_names(path=args.data_path, smiles_columns=args.smiles_columns,
                                    target_columns=args.target_columns, ignore_columns=args.ignore_columns)
    
    logger = create_logger(name=TRAIN_LOGGER_NAME, save_dir=args.save_dir, quiet=args.quiet)
    torch.manual_seed(args.pytorch_seed)

    if os.path.exists(train_pt):
        train_dataset = torch.load(train_pt)
    else:
        train_dataset = get_data(path=args.data_path,
                                args=args,
                                logger=logger,
                                skip_none_targets=True,
                                data_weights_path=args.data_weights_path)
    
        validate_dataset_type(train_dataset, dataset_type=args.dataset_type)
        torch.save(train_dataset, train_pt)

    if os.path.exists(val_pt):
        val_dataset = torch.load(val_pt)
    else:
        val_dataset = get_data(path=args.separate_val_path,
                        args=args,
                        features_path=args.separate_val_features_path,
                        atom_descriptors_path=args.separate_val_atom_descriptors_path,
                        bond_descriptors_path=args.separate_val_bond_descriptors_path,
                        phase_features_path=args.separate_val_phase_features_path,
                        constraints_path=args.separate_val_constraints_path,
                        smiles_columns=args.smiles_columns,
                        loss_function=args.loss_function,
                        logger=logger)
        torch.save(val_dataset, val_pt)

    if os.path.exists(test_pt):
        test_dataset = torch.load(test_pt)
    else:
        test_dataset = get_data(path=args.separate_test_path,
                            args=args,
                            features_path=args.separate_test_features_path,
                            atom_descriptors_path=args.separate_test_atom_descriptors_path,
                            bond_descriptors_path=args.separate_test_bond_descriptors_path,
                            phase_features_path=args.separate_test_phase_features_path,
                            constraints_path=args.separate_test_constraints_path,
                            smiles_columns=args.smiles_columns,
                            loss_function=args.loss_function,
                            logger=logger)
        torch.save(test_dataset, test_pt)

    if args.atom_descriptors == 'descriptor':
        args.atom_descriptors_size = train_dataset.atom_descriptors_size()
    elif args.atom_descriptors == 'feature':
        args.atom_features_size = train_dataset.atom_features_size()
        set_extra_atom_fdim(args.atom_features_size)
    if args.bond_descriptors == 'descriptor':
        args.bond_descriptors_size = train_dataset.bond_descriptors_size()
    elif args.bond_descriptors == 'feature':
        args.bond_features_size = train_dataset.bond_features_size()
        set_extra_bond_fdim(args.bond_features_size)

    if args.dataset_type == 'classification':
        train_class_sizes = get_class_sizes(train_dataset, proportion=False)
        args.train_class_sizes = train_class_sizes
    
    if args.features_scaling:
        features_scaler = train_dataset.normalize_features(replace_nan_token=0)
        val_dataset.normalize_features(features_scaler)
        test_dataset.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.atom_descriptor_scaling and args.atom_descriptors is not None:
        atom_descriptor_scaler = train_dataset.normalize_features(replace_nan_token=0, scale_atom_descriptors=True)
        val_dataset.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
        test_dataset.normalize_features(atom_descriptor_scaler, scale_atom_descriptors=True)
    else:
        atom_descriptor_scaler = None

    if args.bond_descriptor_scaling and args.bond_descriptors is not None:
        bond_descriptor_scaler = train_dataset.normalize_features(replace_nan_token=0, scale_bond_descriptors=True)
        val_dataset.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
        test_dataset.normalize_features(bond_descriptor_scaler, scale_bond_descriptors=True)
    else:
        bond_descriptor_scaler = None

    args.train_data_size = len(train_dataset)
    args.features_size = train_dataset.features_size()

    if args.dataset_type == 'regression':
        if args.is_atom_bond_targets:
            scaler = None
            atom_bond_scaler = train_dataset.normalize_atom_bond_targets()
        else:
            scaler = train_dataset.normalize_targets()
            atom_bond_scaler = None
    else:
        args.spectra_phase_mask = None
        scaler = None
        atom_bond_scaler = None

    train_loader = MoleculeDataLoader(dataset=train_dataset, batch_size=args.batch_size, class_balance=args.class_balance, num_workers=args.num_workers, seed=args.seed, shuffle=True)
    val_loader = MoleculeDataLoader(dataset=val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_loader = MoleculeDataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_targets = test_dataset.targets()

    # Get loss function
    loss_func = get_loss_func(args)

    model = MoleculeModel(args)
    model.to(device)

    # Optimizers
    optimizer = build_optimizer(model, args)

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)

    n_iter = 0

    # Run training
    for epoch in range(1, args.epochs+ 1): ## wandb.config.epochs + 1
        n_iter = train(
            model=model,
            data_loader=train_loader,
            loss_func=loss_func,
            optimizer=optimizer,
            scheduler=scheduler,
            args=args,
            n_iter=n_iter,
            atom_bond_scaler=atom_bond_scaler,
            logger=logger,
            writer=None
        )
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        
        val_scores = evaluate(
            model=model,
            data_loader=val_loader,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            scaler=scaler,
            atom_bond_scaler=atom_bond_scaler,
            logger=logger
        )

        test_preds = predict(
            model=model,
            data_loader=test_loader,
            scaler=scaler,
            atom_bond_scaler=atom_bond_scaler
        )

        test_scores = evaluate_predictions(
            preds=test_preds,
            targets=test_targets,
            num_tasks=args.num_tasks,
            metrics=args.metrics,
            dataset_type=args.dataset_type,
            is_atom_bond_targets=args.is_atom_bond_targets,
            gt_targets=test_dataset.gt_targets(),
            lt_targets=test_dataset.lt_targets(),
            logger=logger
        )

        if GLOBAL_ARGS['no_wandb']:
            if type_of_run == "classification":
                print({"epoch": epoch,"validation_mcc": val_scores['mcc'][0], "validation_auc": val_scores['auc'][0],\
                        "validation_acc": val_scores['accuracy'][0], "optimizer": "Adam","learning_rate": scheduler.get_lr()[0],\
                        "test_mcc": test_scores['mcc'][0], "test_auc": test_scores['auc'][0], "test_acc": test_scores['accuracy'][0],\
                        "activation": args.activation,"batch_size": args.batch_size, "drop_out": args.dropout,\
                        "ffn_num_layers" : args.ffn_num_layers, "bias" : args.bias, "depth": args.depth, \
                        "weights_ffn_num_layers": args.weights_ffn_num_layers, "aggregation" : args.aggregation,\
                        "validation_corr":val_scores['mcc'][0],\
                        # "test_confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=test_targets, preds=test_preds)
                        })
            else:
                print({"epoch": epoch,"validation_r2": val_scores['r2'][0], "validation_rmse": val_scores['rmse'][0],\
                        "optimizer": "Adam","learning_rate": scheduler.get_lr()[0],\
                        "test_r2": test_scores['r2'][0], "test_rmse": test_scores['rmse'][0],\
                        "activation": args.activation,"batch_size": args.batch_size, "drop_out": args.dropout,\
                        "ffn_num_layers" : args.ffn_num_layers, "bias" : args.bias, "depth": args.depth, \
                        "weights_ffn_num_layers": args.weights_ffn_num_layers, "aggregation" : args.aggregation,\
                        "validation_corr": val_scores['r2'][0]
                        })

        else:
            if type_of_run == "classification":
                print({"epoch": epoch,"validation_mcc": val_scores['mcc'][0]})
                wandb.log({"epoch": epoch,"validation_mcc": val_scores['mcc'][0], "validation_auc": val_scores['auc'][0],\
                        "validation_acc": val_scores['accuracy'][0], "optimizer": "Adam","learning_rate": scheduler.get_lr()[0],\
                        "test_mcc": test_scores['mcc'][0], "test_auc": test_scores['auc'][0], "test_acc": test_scores['accuracy'][0],\
                        "activation": args.activation,"batch_size": args.batch_size, "drop_out": args.dropout,\
                        "ffn_num_layers" : args.ffn_num_layers, "bias" : args.bias, "depth": args.depth, \
                        "weights_ffn_num_layers": args.weights_ffn_num_layers, "aggregation" : args.aggregation,\
                        "validation_corr":val_scores['mcc'][0], "add_feats" : wandb.config.add_feats,\
                        # "test_confusion_matrix": wandb.plot.confusion_matrix(probs=None, y_true=test_targets, preds=test_preds)
                        })
            else:
                print({"epoch": epoch,"validation_r2": val_scores['r2'][0]})
                wandb.log({"epoch": epoch,"validation_r2": val_scores['r2'][0], "validation_rmse": val_scores['rmse'][0],\
                        "optimizer": "Adam","learning_rate": scheduler.get_lr()[0],\
                        "test_r2": test_scores['r2'][0], "test_rmse": test_scores['rmse'][0],\
                        "activation": args.activation,"batch_size": args.batch_size, "drop_out": args.dropout,\
                        "ffn_num_layers" : args.ffn_num_layers, "bias" : args.bias, "depth": args.depth, \
                        "weights_ffn_num_layers": args.weights_ffn_num_layers, "aggregation" : args.aggregation,\
                        "validation_corr": val_scores['r2'][0], "add_feats" : wandb.config.add_feats,\
                        })
                
    if not GLOBAL_ARGS['no_wandb']:
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('label', type=str, help='target name')
    parser.add_argument('-offline', help='go offline', action='store_true')
    parser.add_argument('-no_wandb', help='go without wandb', action='store_true')
    parser.add_argument('-add_feats', type=str, help='add feats')
    
    args = parser.parse_args()

    GLOBAL_ARGS['label'] = args.label
    GLOBAL_ARGS['offline'] = args.offline
    GLOBAL_ARGS['no_wandb'] = args.no_wandb

    WANDB_API_KEY = '54c05c1e175ce6a74077275f4fde516fa66ae250'
    os.environ['WANDB_API_KEY'] = WANDB_API_KEY
    
    add_feats = args.add_feats.split(',')
    
    sweep_config = {
        'method': 'bayes'
    }
    metric = {
        'name': 'validation_corr',
        'goal': 'maximize'
    }
    parameters_dict = {
        'ffn_num_layers':{
            'distribution' : 'q_uniform',
            'min': 1,
            'max': 3,
            'q' : 1
        },
        'hidden_size':{
            'distribution' : 'q_uniform',
            'min': 300,
            'max': 2400,
            'q' : 100
        },
        'ffn_hidden_size':{
            'distribution' : 'q_uniform',
            'min': 300,
            'max': 2400,
            'q' : 100
        },
        'depth':{
            'distribution' : 'q_uniform',
            'min': 2,
            'max': 6,
            'q' : 1
        },
        'dropout':{
            'distribution' : 'q_uniform',
            'min': 0,
            'max': 0.4,
            'q' : 0.05
        },
        'add_feats':{
            'values': add_feats
            #'values': ['mordred', 'mordred_powertransform', 'mordred_reduced' ]
        },
    }

    early_terminate = {
        'type': 'hyperband',
        's' : 2,
        'eta' : 3,
        'max_iter': 27
    }

    sweep_config['metric'] = metric
    sweep_config['parameters'] = parameters_dict
    sweep_config['early_terminate'] = early_terminate
    
    if args.offline:
        os.environ['WANDB_MODE'] = "offline"
    
    if args.no_wandb:
        run_Chemprop() # TODO: need to update arguments for Chemprop
        pass
    
    else:
        wandb.login()
        sweep_id = wandb.sweep(sweep_config, project="Chemprop-toxCSM-hypopt-{}".format(args.label))
        wandb.agent(sweep_id, function=run_Chemprop)
     
   
    
