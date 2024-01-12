import wandb, glob, os, sys, pickle
from tqdm import tqdm
import polars as pl
reg_list = ['bbb_cns','bioconcF','bp','caco2','caco2_logPaap','cl','fdamdd_reg','fm_reg','fu','hydrationE','lc50','lc50dm','ld50','logbcf','logd','logp','logs','logvp','mdck','mp','pka','pkb','ppb','pyriformis_reg','rat_acute_reg','rat_chronic','skin_permeability','vd']
WANDB_API_KEY = '54c05c1e175ce6a74077275f4fde516fa66ae250'
os.environ['WANDB_API_KEY'] = WANDB_API_KEY

def main(target_dir):
	endpoint_name = os.path.abspath(target_dir).split('/')[-1]

	## WANDB
	metric, wandb_pl = fetch_Wandb_results(endpoint_name)

	pkl_list = glob.glob(os.path.join(target_dir,'*.pkl'))
	pkl_list.sort()

	doesnt_exist_list = list()
	does_exist_list = list()

	for pkl in pkl_list:
		_Trials = pickle.load(open(pkl, 'rb'))
		_mean_score = _Trials.best_trial['result']['mean_score']
		_num_params = _Trials.best_trial['result']['num_params']
		_seed = _Trials.best_trial['result']['seed']
		_hyperparams = _Trials.best_trial['result']['hyperparams']

		# check_condition = wandb_pl.filter((pl.col("seed") == str(_seed)) & (pl.col(f"{metric}_mean") == str(_mean_score)) & (pl.col("num_params") == str(_num_params)))
		check_condition = wandb_pl.filter((pl.col("seed") == str(_seed)) & (pl.col("num_params") == str(_num_params)))
		if len(check_condition) == 0:
			doesnt_exist_list.append(pkl)
			# run = wandb.init(project="DEEPPK-hypopt-{}".format(endpoint_name),\
			# notes="Manually added",\
			# config = _hyperparams) 
			# wandb.log({'seed': _seed,'loss': str(_Trials.best_trial['result']['loss']), 'num_params': _num_params,\
			#   f'{metric}_mean': str(_Trials.best_trial['result']['mean_score']), f'{metric}_std': str(_Trials.best_trial['result']['std_score'])})
			# wandb.finish()
			# print(check_condition)
			# print(check_condition[['seed', 'num_params']])

			# break
		else:
			does_exist_list.append(pkl)
			# print(check_condition)
			# print(check_condition[['seed', 'num_params']])
			# break
	
	print(len(doesnt_exist_list))
	print(doesnt_exist_list)
	print(len(does_exist_list),len(wandb_pl))

	if len(doesnt_exist_list) > 0:
		if len(wandb_pl) -20 <= len(does_exist_list) <=len(wandb_pl) +20:
			for pkl in pkl_list:
				_Trials = pickle.load(open(pkl, 'rb'))
				_mean_score = _Trials.best_trial['result']['mean_score']
				_num_params = _Trials.best_trial['result']['num_params']
				_seed = _Trials.best_trial['result']['seed']
				_hyperparams = _Trials.best_trial['result']['hyperparams']

				check_condition = wandb_pl.filter((pl.col("seed") == str(_seed)) & (pl.col("num_params") == str(_num_params)))
				if len(check_condition) == 0:
					run = wandb.init(project="DEEPPK-hypopt-{}".format(endpoint_name),\
					notes="Manually added",\
					config = _hyperparams) 
					wandb.log({'seed': _seed,'loss': str(_Trials.best_trial['result']['loss']), 'num_params': _num_params,\
					f'{metric}_mean': str(_Trials.best_trial['result']['mean_score']), f'{metric}_std': str(_Trials.best_trial['result']['std_score'])})
					wandb.finish()
		else:
			print(f"{endpoint_name} NEEDS TO BE CHECKED {len(does_exist_list)}, {len(wandb_pl)}")
	else:
		print(f"{endpoint_name} DOESN'T NEED TO BE UPDATED.")



# Bring all results from Wandb First and make it as pandas table
def fetch_Wandb_results(target):
	# target = kwargs.target

	summary_pd = pl.DataFrame()
	best_row = dict()
	api = wandb.Api()
	metric = str()

	title = f'DEEPPK-hypopt-{target}'

	if target in reg_list:
		metric = 'r2'
	else:
		metric = 'mcc'

	runs = api.runs(title)

	for run in tqdm(runs):
		try:
			_pd = pl.DataFrame({f'{metric}_mean': float(run.summary.get(f'{metric}_mean')),\
								f'{metric}_std': float(run.summary.get(f'{metric}_std')),
								'run_name': str(run.name),\
								'run_id': str(run.id),\
								'index' : str(run.name),\
								'seed': int(run.summary.get('seed')),
								'num_params': int(run.summary.get('num_params'))
								})
			
			_pd = pl.concat([_pd, pl.DataFrame(run.config)], how='horizontal')
			_pd = _pd.select([pl.all().cast(pl.Utf8)])
			summary_pd = pl.concat([summary_pd,_pd], how='vertical')
			
			# break
		except Exception as e:
			pass
		
	# summary_pd = summary_pd.with_columns(pl.col(f'{metric}_mean').cast(pl.Float64))
	# summary_pd_filtered = summary_pd.filter(summary_pd[f'{metric}_mean']>=0)
	# print(summary_pd[['seed','num_params']])
	return (metric, summary_pd)

if __name__ == '__main__':
	target_dir = sys.argv[1]
	main(target_dir)
