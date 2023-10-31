import wandb, argparse, json, os
import polars as pl
pl.Config.set_fmt_float("full")
from tqdm import tqdm

def fetch_result(kwargs):
	print(kwargs)
	if isinstance(kwargs, dict):
		target = kwargs['target']
		reg = kwargs['reg']
		title = kwargs['title']
		run_name = kwargs['run_name']

	else:
		target = kwargs.target
		reg = kwargs.reg
		title = kwargs.title
		run_name = kwargs.run_name

	summary_pd = pl.DataFrame()
	best_row = dict()
	wandb.login(key='54c05c1e175ce6a74077275f4fde516fa66ae250')
	api = wandb.Api()
	metric = str()

	title = f'{title}-{target}'

	if reg:
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
								'index' : str(run.name)})
			_pd = pl.concat([_pd, pl.DataFrame(run.config)], how='horizontal')
			_pd = _pd.select([pl.all().cast(pl.Utf8)])
			summary_pd = pl.concat([summary_pd,_pd], how='vertical')
		except Exception as e:
			pass
	
	summary_pd = summary_pd.with_columns(pl.col(f'{metric}_mean').cast(pl.Float64))
	summary_pd_filtered = summary_pd.filter(summary_pd[f'{metric}_mean']>=0)
		
	if str(run_name) == 'None':
		summary_pd_filtered = summary_pd_filtered.sort(f'{metric}_mean', descending=True)

		best_row = summary_pd_filtered.row(0, named=True)
		
		best_run = api.run(f"{title}/{best_row['run_id']}")

	else:
		run_id = summary_pd_filtered.filter(summary_pd_filtered['run_name'].str.contains(f'{run_name}')).row(0,named=True)['run_id']
		best_run = api.run(f"{title}/{run_id}")
		_pd = pl.DataFrame({f'{metric}_mean': float(best_run.summary.get(f'{metric}_mean')),\
							f'{metric}_std': float(best_run.summary.get(f'{metric}_std')),
							'run_name': str(best_run.name),\
							'run_id': str(best_run.id),\
							'index' : str(best_run.name)})

		_pd = pl.concat([_pd, pl.DataFrame(best_run.config)], how='horizontal')
		best_row.update(_pd.to_dict(as_series=False))
		best_row = {key: str(value_list[0]) for key, value_list in best_row.items()}

	mordred_count = len([each for each in json.load(best_run.file("wandb-metadata.json").download(root=target, replace=True))['args'] if 'features_path' in each])

	if mordred_count > 2:
		best_row.update({'mordred':'True'})
	else:
		best_row.update({'mordred':'False'})

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

	output_config['init_lr'] = float(best_row['max_lr']) * float(best_row['init_lr_ratio'])
	output_config['final_lr'] = float(best_row['max_lr']) * float(best_row['final_lr_ratio'])

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(output_name, 'w') as fp:
		json.dump(best_row, fp)

	#with open(output_config_name, 'w') as fcp:
	#	json.dump(output_config, fcp)

	return best_row


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="A simple argument parser")
	parser.add_argument("target", type=str, help="A Target name")
	parser.add_argument("-reg", action='store_true')
	parser.add_argument("-title", type=str, default='biosig/ADMETLAB2-hypopt')
	parser.add_argument("-run_name", type=str, default=None)

	args = parser.parse_args()
	print(fetch_result(args))
