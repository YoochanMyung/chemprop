import wandb, glob, os, sys, pickle, argparse, copy
from tqdm import tqdm
import polars as pl
reg_list = ['bbb_cns','bioconcF','bp','caco2','caco2_logPaap','cl','fdamdd_reg','fm_reg','fu','hydrationE','lc50','lc50dm','ld50','logbcf','logd','logp','logs','logvp','mdck','mp','pka','pkb','ppb','pyriformis_reg','rat_acute_reg','rat_chronic','skin_permeability','vd']
WANDB_API_KEY = '54c05c1e175ce6a74077275f4fde516fa66ae250'
os.environ['WANDB_API_KEY'] = WANDB_API_KEY
cwd = os.getcwd()

def main(args):
	endpoint_name = args.endpoint
	wandb_project_name = args.project_name
	# print(endpoint_name, wandb_project_name, cwd)
	target_dir = os.path.join(cwd, endpoint_name)
	## WANDB
	
	wandb_pl = fetch_Wandb_results(wandb_project_name, endpoint_name)

	pkl_list = glob.glob(os.path.join(target_dir,'*.pkl'))
	pkl_list.sort()

	doesnt_exist_list = list()
	does_exist_list = list()

	## Check ##
	for pkl in pkl_list:
		_Trials = pickle.load(open(pkl, 'rb'))
		_seed = _Trials.best_trial['result']['seed']
		check_condition = wandb_pl.filter(pl.col("seed") == str(_seed))

		if len(check_condition) == 0:
			doesnt_exist_list.append(pkl)
		else:
			does_exist_list.append(pkl)
	print(f"Doesn't: {len(doesnt_exist_list)} , Does: {len(does_exist_list)}, Wandb: {len(wandb_pl}")

	if len(doesnt_exist_list) > 0:
		for pkl in doesnt_exist_list:
			_scores = dict()
			_upload_dict = dict()

			_Trials = pickle.load(open(pkl, 'rb'))
			_seed = _Trials.best_trial['result']['seed']
			_num_params = _Trials.best_trial['result']['num_params']
			_hyperparams = _Trials.best_trial['result']['hyperparams']

			_hostname = glob.glob(os.path.join(os.path.dirname(pkl),f'trial_seed_{_seed}/fold_0/model_0/event*'))[0].split('.')[-1]
			_test_scores_csv = os.path.join(os.path.dirname(pkl), f'trial_seed_{_seed}','test_scores.csv')
			_test_scores_dict = pl.read_csv(_test_scores_csv).to_dict(as_series=False)

			if not endpoint_name in reg_list:
				for each_class_metric in ['mcc','auc','f1','accuracy']:
					_scores[f'{each_class_metric}_mean'] = _test_scores_dict[f'Mean {each_class_metric}'][0]
					_scores[f'{each_class_metric}_std'] = _test_scores_dict[f'Standard deviation {each_class_metric}'][0]
			else:
				for each_reg_metric in ['mae','mse','rmse','r2']:
					_scores[f'{each_reg_metric}_mean'] = _test_scores_dict[f'Mean {each_reg_metric}'][0]
					_scores[f'{each_reg_metric}_std'] = _test_scores_dict[f'Standard deviation {each_reg_metric}'][0]
			
			_upload_dict.update(_scores)
			_upload_dict['hostname'] = _hostname
			_upload_dict['seed'] = _seed
			_upload_dict['num_params'] = _num_params
			_upload_dict['loss'] = str(_Trials.best_trial['result']['loss'])

			check_condition = wandb_pl.filter(pl.col("seed") == str(_seed))

			if len(check_condition) == 0:
				run = wandb.init(project=f"{wandb_project_name}-{endpoint_name}",\
				notes="Manually added",\
				config = _hyperparams) 
				wandb.log(_upload_dict)
				wandb.finish()
	else:
		print(f"{endpoint_name} DOESN'T NEED TO BE UPDATED.")

# Bring all results from Wandb First and make it as pandas table
def fetch_Wandb_results(wandb_project_name, target):
	summary_pd = pl.DataFrame()
	api = wandb.Api()
	title = f'{wandb_project_name}-{target}'
	runs = api.runs(title)

	for run in tqdm(runs):
		try:
			summary_dict = dict(run.summary)
			summary_dict.pop('_timestamp')
			summary_dict.pop('_runtime')
			summary_dict.pop('_step')
			summary_dict.pop('_wandb')

			_pd = pl.concat([pl.DataFrame(summary_dict), pl.DataFrame(run.config)], how='horizontal')
			_pd = _pd.select([pl.all().cast(pl.Utf8)])

			summary_pd = pl.concat([summary_pd,_pd], how='diagonal')
		except KeyError as e:
			pass
		
	return summary_pd

# Bring all results from Wandb First and make it as pandas table
def remove_non_hostname_Wandb_results(wandb_project_name, target):
	summary_pd = pl.DataFrame()
	api = wandb.Api()
	title = f'{wandb_project_name}-{target}'
	runs = api.runs(title)

	for run in tqdm(runs):
		# summary_dict = dict(run.summary)
		# print(summary_dict['hostname'])
		_hostname = run.summary.get('hostname')
		if not _hostname:
			print(_hostname, run)
			run.delete()
		
	return summary_pd
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='This script is for updating offline Hypopt jobs to Wandb.')
	parser.add_argument('project_name', type=str, help='Specify a project name of Wandb.')
	parser.add_argument('endpoint', type=str, help='Provide an Endpoint name.')
	args = parser.parse_args()
	main(args)
	# fetch_Wandb_results('DEEPPK-pdCSM-hypopt', 'rat_acute_class')
	# remove_non_hostname_Wandb_results(args.project_name, args.endpoint)
