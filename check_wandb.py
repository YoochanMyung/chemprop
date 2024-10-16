import wandb, argparse, json, os
# import polars as pl
# pl.Config.set_fmt_float("full")
from tqdm import tqdm
import pandas as pd

admetlab2_endpoints = ["ames","bbb_logbb","carcinogenicity","cyp1a2_inhibitor","cyp1a2_substrate","cyp2c19_inhibitor","cyp2c19_substrate","cyp2c9_inhibitor","cyp2c9_substrate","cyp2d6_inhibitor","cyp2d6_substrate","cyp3a4_inhibitor","cyp3a4_substrate","dili","eye_corrosion","eye_irritation","f20","f30","fdamdd_class","h_ht","herg","hia","nr_ahr","nr_ar","nr_ar_lbd","nr_aromatase","nr_er","nr_er_lbd","nr_ppar_gamma","pgp_inhibitor","pgp_substrate","rat_acute_class","respiratory_tox","skin_sens","sr_are","sr_atad5","sr_hse","sr_mmp","sr_p53","t0.5","bioconcF","caco2","cl","fm_reg","fu","lc50dm","logd","logp","logs","mdck","ppb","pyriformis_reg","vd"]
toxcsm_endpoints = ["ames","avian_tox","bee_tox","biodegradation","carcinogenicity","dili","eye_corrosion","eye_irritation","fm_class","h_ht","herg1","herg2","micronucleus_tox","nr_ahr","nr_ar","nr_ar_lbd","nr_aromatase","nr_er","nr_er_lbd","nr_gr","nr_ppar_gamma","nr_tr","pyriformis_class","skin_sens","sr_are","sr_atad5","sr_hse","sr_mmp","sr_p53","rat_chronic","fm_reg","pyriformis_reg","rat_acute_reg","fdamdd_reg"]
deeppk_endpoints = ["ames","avian_tox","bbb_cns","bbb_logbb","bcrp","bee_tox","bioconcF","biodegradation","bp","caco2_reg","carcinogenicity","cl","crustacean","cyp1a2_inhibitor","cyp1a2_substrate","cyp2c19_inhibitor","cyp2c19_substrate","cyp2c9_inhibitor","cyp2c9_substrate","cyp2d6_inhibitor","cyp2d6_substrate","cyp3a4_inhibitor","cyp3a4_substrate","dili","eye_corrosion","eye_irritation","f20","fdamdd_reg","fu","h_ht","herg","hia","hydrationE","fm_reg","lc50dm","logd","logp","logs","logvp","mdck","micronucleus_tox","mp","nr_ahr","nr_ar","nr_ar_lbd","nr_aromatase","nr_er","nr_er_lbd","nr_gr","nr_ppar_gamma","nr_tr","oatp1b1","oatp1b3","ob","oct2","pgp_inhibitor","pgp_substrate","pka","pkb","ppb","pyriformis_reg","rat_acute_reg","rat_chronic","respiratory_tox","skin_permeability","skin_sens","sr_are","sr_atad5","sr_hse","sr_mmp","sr_p53","t0.5","vd"]
reg_list = ['bbb_cns','bioconcF','bp','caco2','caco2_logPaap','cl','fdamdd_reg','fm_reg','fu','hydrationE','lc50','lc50dm','ld50','logbcf','logd','logp','logs','logvp','mdck','mp','pka','pkb','ppb','pyriformis_reg','rat_acute_reg','rat_chronic','skin_permeability','vd','bbb(lobbb)','caco2_reg']

def fetch_result(kwargs):
	print(kwargs)
	if isinstance(kwargs, dict):
		target = kwargs['target']
		# reg = kwargs['reg']
		project = kwargs['project']
		# run_name = kwargs['run_name']

	else:
		target = kwargs.target
		# reg = kwargs.reg
		project = kwargs.project
		# run_name = kwargs.run_name

	summary_pd = pd.DataFrame()
	best_row = dict()
	wandb.login(key='54c05c1e175ce6a74077275f4fde516fa66ae250')
	api = wandb.Api()
	metric = str()
	project_title = str()

	# project = f'{project}-{target}'
	if project == 'deeppk':
		project_title = f'new-DEEPPK-hypopt-{target}'
	else:
		project_title = f'new-DEEPPK-{project}-hypopt-{target}'
	
	if target in reg_list:
		metric = 'r2'
	else:
		metric = 'mcc'

	runs = api.runs(project_title)
	# import pdb;pdb.set_trace()
	for run in tqdm(runs):
		# try:
		_temp_dict = dict()
		_temp_dict.update(dict(run.summary))
		_temp_dict.update(run.config)
		_temp_dict['run_name'] = str(run.name)
		_temp_dict['run_id'] = str(run.id)
		# _temp_dict['index'] = str(run.name)
		_pd = pd.DataFrame(_temp_dict, index=[run.name])
		summary_pd = pd.concat([summary_pd,_pd])
		# except Exception as e:
		# 	pass
	import numpy as np
	summary_pd = summary_pd.replace(np.nan, 'None')
	# import pdb;pdb.set_trace()
	summary_pd_filtered = summary_pd.query(f'{metric}_mean >= 0')
	# print(summary_pd_filtered)
	summary_pd_filtered.sort_values(by=f'{metric}_mean', ascending=False, inplace=True)
	best_row = summary_pd_filtered.iloc[0]
	# if str(run_name) == 'None':
		# summary_pd_filtered.sort_values(by=f'{metric}_mean', ascending=False, inplace=True)
		# best_row = summary_pd_filtered.iloc[0]

	# else:
		# TODO: Needs to be changed for Pandas
		# run_id = summary_pd_filtered.filter(summary_pd_filtered['run_name'].str.contains(f'{run_name}')).row(0,named=True)['run_id']
		# best_run = api.run(f"{project}/{run_id}")
		# _pd = pl.DataFrame({f'{metric}_mean': float(best_run.summary.get(f'{metric}_mean')),\
		# 					f'{metric}_std': float(best_run.summary.get(f'{metric}_std')),
		# 					'run_name': str(best_run.name),\
		# 					'run_id': str(best_run.id),\
		# 					'index' : str(best_run.name)})

		# _pd = pl.concat([_pd, pl.DataFrame(best_run.config)], how='horizontal')
		# best_row.update(_pd.to_dict(as_series=False))
		# best_row = {key: str(value_list[0]) for key, value_list in best_row.items()}
		# pass

	#output_name = os.path.abspath(f'./{target}/{best_run.name}/{target}_best.json')
	output_name = os.path.abspath(f'{target}_best.json')
	#output_config_name = os.path.abspath(f'./{target}/{best_run.name}/{target}_best_config.log')

	output_dir = os.path.dirname(output_name)
	output_config = dict()

	for k,v in best_row.items():
		if k in ['batch_size', 'depth', 'ffn_hidden_size', 'ffn_num_layers', 'hidden_size', 'warmup_epochs']:
			output_config[k] = int(v)

		elif k in ['final_lr', 'max_lr', 'init_lr', 'aggregation_norm', 'dropout']:
			output_config[k] = float(v)

		else:
			output_config[k] = str(v)

	# output_config['init_lr'] = float(best_row['max_lr']) * float(best_row['init_lr_ratio'])
	# output_config['final_lr'] = float(best_row['max_lr']) * float(best_row['final_lr_ratio'])
	best_row['init_lr'] = float(best_row['max_lr']) * float(best_row['init_lr_ratio'])
	best_row['final_lr'] = float(best_row['max_lr']) * float(best_row['final_lr_ratio'])

	# if not os.path.exists(output_dir):
	# 	os.makedirs(output_dir)

	# print(best_row)
	# with open(output_name, 'w') as fp:
	# 	json.dump(best_row, fp)

	#with open(output_config_name, 'w') as fcp:
	#	json.dump(output_config, fcp)
	return best_row.to_dict()


def run_all(kwargs):
	if isinstance(kwargs, dict):
		# target = kwargs['target']
		project = kwargs['project']


	else:
		# target = kwargs.target
		project = kwargs.project

	if project == 'deeppk':
		target_list = deeppk_endpoints
	elif project == 'toxcsm':
		target_list = toxcsm_endpoints
	else:
		target_list = admetlab2_endpoints

	all_results = pd.DataFrame()
	# for each_target in target_list:
	for each_target in ['cyp2c19_substrate']:
		kwargs.target = each_target
		all_results = pd.concat([all_results,pd.DataFrame(fetch_result(kwargs),index=[each_target])])
		# break
	# print(all_results)
	if project == 'deeppk':
		output_name = f'all_results_new-DEEPPK-hypopt.csv'
	else:
		output_name = f'all_results_new-DEEPPK-{project}-hypopt.csv'
	all_results.to_csv(output_name)
	# pass


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="A simple argument parser")
	parser.add_argument("project", type=str, choices=['deeppk','admetlab2','toxcsm'])
	# parser.add_argument("target", type=str, help="A Target name")
	# parser.add_argument("-reg", action='store_true')
	# parser.add_argument("-run_name", type=str, default=None)

	args = parser.parse_args()
	# print(fetch_result(args))
	run_all(args)

#cyp2c19_substrate