import argparse
import os
import chemprop
import simplejson as json
from check_wandb import fetch_result
import pandas as pd
import sys, pickle
#sys.path.append('/home/ymyung/projects/deeppk/src/chemprop') # Baker pc
#sys.path.append('/clusterdata/uqymyung/src/chemprop') # WIENER
sys.path.append('/home/uqymyung/src/chemprop') # friday

def check_categorical(input_csv):
	input_pd = pd.read_csv(input_csv)
	components = set(input_pd['label'].to_list())
	if len(components) > 5:
		return False
	else:
		return True

def run(kwargs):
	target_data = kwargs.data
    # csv_dir = '/home/ymyung/projects/deeppk/2_ML_running/2_Chemprop/5_using_other_DBs/dataset/admetlab2' # Baker pc
	# csv_dir = f'/clusterdata/uqymyung/uqymyung/projects/deeppk/1_dataset/{kwargs.data}' # WIENER
	# csv_dir = f'/home/ymyung/Projects/deeppk/data/{target_data}' # for bio21
	csv_dir = f'/home/uqymyung/scratch/projects/deeppk/dataset/' # for friday
	if not kwargs.title and target_data == 'toxcsm':
		title = 'biosig/toxCSM-hypopt'
	else:
		title = kwargs.title

	if not kwargs.config:
		best_run = fetch_result({'target': kwargs.endpoint, 'reg': kwargs.reg, 'run_name': kwargs.run_name, 'title': title})
		run_name = best_run['run_name']
	else:
		best_run = 'local_run'

	print("Working on: {}".format(kwargs.endpoint))

	if kwargs.run_name:
		save_dir = os.path.join(kwargs.save_path, kwargs.endpoint, kwargs.run_name)
	else:
		save_dir = os.path.join(kwargs.save_path, kwargs.endpoint)

	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	if not kwargs.config:
		result_json = json.load(open(os.path.join(f'{kwargs.save_path}/{kwargs.endpoint}_best.json'), 'r')) # Wandb  WIENER
	else:
		result_json = pickle.load(open(kwargs.config, 'rb')) # local

	result_json = {k: str(v) for k, v in result_json.items()}

	init_lr = float(result_json['max_lr']) * float(result_json['init_lr_ratio'])
	final_lr = float(result_json['max_lr']) * float(result_json['final_lr_ratio'])

	arguments = [
	'--smiles_columns','smiles_standarized',
	'--target_columns','label',
	## Input paths
	'--data_path', os.path.join(csv_dir, f'{kwargs.endpoint}_train.csv'),
	'--separate_val_path', os.path.join(csv_dir, f'{kwargs.endpoint}_val.csv') if target_data == 'admetlab2' else os.path.join(csv_dir, f'{kwargs.endpoint}_test.csv'),
	'--separate_test_path', os.path.join(csv_dir, f'{kwargs.endpoint}_test.csv'),
	'--save_dir', save_dir,
	## Model arguments
	'--depth', result_json['depth'],
	'--init_lr', str(init_lr),
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
	'--aggregation_norm', str(int(float(result_json["aggregation_norm"]))),
	'--num_folds','3',
	'--save_preds',
	'--num_workers','8']

	if kwargs.reg:
		arguments = arguments + [
		'--dataset_type', 'regression',
		'--metric','r2',
		'--extra_metrics', 'rmse','mse','mae',
		]
	else:
		arguments = arguments + [
		'--dataset_type', 'classification',
		'--metric','mcc',
		'--extra_metrics', 'f1','auc','accuracy',
		]

	if kwargs.mordred or result_json.get('mordred') == 'True':
		arguments = arguments + [
		'--features_path', os.path.join(csv_dir, 'mordred', f'{kwargs.endpoint}_train.csv'),
		'--separate_val_features_path', os.path.join(csv_dir, 'mordred',  f'{kwargs.endpoint}_val.csv' if target_data == 'admetlab2' else f'{kwargs.endpoint}_test.csv'),
		'--separate_test_features_path', os.path.join(csv_dir, 'mordred', f'{kwargs.endpoint}_test.csv'),
		]

	args = chemprop.args.TrainArgs().parse_args(arguments)
	mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='ex) python build_hypopt_model.py run . -run_name silvery-dust-1001')
	parser.add_argument("data", help="Choose dataset",type=str, choices=['admetlab2','toxcsm','deeppk'])
	parser.add_argument("endpoint", help="Choose endpoint name",type=str)
	parser.add_argument("save_path", help="Choose dir path for save",type=str)
	parser.add_argument("-config", type=str, help="Provide a config.json rather than searhcing Wandb.", default=None)
	parser.add_argument("-reg", action='store_true', help="")
	parser.add_argument("-mordred", action='store_true', help="Run type between regression and classification", default=False)
	parser.add_argument("-run_name",type=str, default=None)
	parser.add_argument("-title",type=str, default=None)

	args = parser.parse_args()
	run(args)
