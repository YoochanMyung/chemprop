import wandb, argparse
import pandas as pd
import json

from tqdm import tqdm

def fetch_result(kwargs):
	# kwargs = vars(kwargs)
	target = kwargs['target']
	reverse = kwargs['r']
	summary_pd = pd.DataFrame()
	api = wandb.Api()	
	runs = api.runs(f"biosig/ADMETLAB2-hypopt-{target}")
	# runs = api.runs(f"biosig/ADMETLAB2-hypopt-ames")
	# print(runs)
	# import pdb; pdb.set_trace()
	for run in tqdm(runs):

		try:
			_pd = pd.DataFrame({'mcc_mean': float(run.summary.get('mcc_mean')),\
								'mcc_std': float(run.summary.get('mcc_std')),
								'run_name': str(run.name),\
								'run_id': str(run.id)}, index=[run.name]
								)
			_pd = _pd.join(pd.DataFrame(run.config, index=[run.name]))
		except:
			pass
			
		summary_pd = pd.concat([summary_pd,_pd])			
	summary_pd.dropna(inplace=True)
	summary_pd = summary_pd.query('mcc_mean >= 0')

	if not reverse:
		summary_pd.sort_values(by='mcc_mean', ascending=False, inplace=True)
	else:
		summary_pd.sort_values(by='mcc_mean', ascending=True, inplace=True)

	best_run = api.run(f"biosig/ADMETLAB2-hypopt-{target}/{summary_pd.iloc[0].run_id}")
	mordred_count = len([each for each in json.load(best_run.file("wandb-metadata.json").download(replace=True))['args'] if 'mordred' in each])

	if mordred_count > 3:
		summary_pd.loc[summary_pd.iloc[0].run_name, 'mordred'] = "True"
	else:
		summary_pd.loc[summary_pd.iloc[0].run_name, 'mordred'] = "False"

	# print(summary_pd.iloc[0])
	summary_pd.iloc[0].to_json(f'{target}_best.json')

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="A simple argument parser")
	parser.add_argument("target", type=str, help="A Target name")
	parser.add_argument("--r", action='store_true')

	args = parser.parse_args()
	# print(args)
	fetch_result(args)