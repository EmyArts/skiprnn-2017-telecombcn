from sklearn.model_selection import ParameterGrid
from imdb import SkipRNN
from imdb import get_embedding_dicts
import tensorflow as tf
import argparse
import os

command_configs = {
	'learning_rate': [0.01, 0.001, 0.0001],
	'batch_size': [32, 64],
	'hidden_units': [32, 64, 96],
	'cost_per_sample': [0.005, 0.001, 0.0001],
	'surprisal_cost': [0.1, 0.005]
}
# command_configs = {
# 	'learning_rate': [0.01],
# 	'batch_size': [32],
# 	'hidden_units': [32],
# 	'cost_per_sample': [0.005],
# 	'surprisal_cost': [0.1]
# }

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--exp_id", type=int, help="id of the specific run")
	parser.add_argument("--tot_exps", type=int, default=48, help="The total amount of parallel experiments")

	args = parser.parse_args()
	exp_id = args.exp_id
	tot_exps = args.tot_exps

	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		try:
			# Currently, memory growth needs to be the same across GPUs
			for gpu in gpus:
				tf.config.experimental.set_memory_growth(gpu, True)
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Memory growth must be set before GPUs have been initialized
			print(e)

	embedding_dict, probs_dict = get_embedding_dicts(50)
	for idx, params in enumerate(ParameterGrid(command_configs)):
		if idx % tot_exps == exp_id:
			csv_name = 'hu' + str(params['hidden_units']) + '_bs' + str(params['batch_sise']) + '_lr' + str(
				params['learning_rate']) + \
					   '_b' + str(params['cost_per_sample']) + '_s' + str(params['surprisal_cost']) + '.csv'
			if not os.path.exists('../completed_csv' + csv_name):
				params['epochs'] = 12
				params['early_stopping'] = 'yes'
				params['folder'] = '../EXP' + exp_id + '_LR' + str(params['learning_rate']) + '_BS' + str(
					params['batch_size']) + \
								   '_HU' + str(params['hidden_units']) + '_CPS' + str(
					params['cost_per_sample']) + '_SC' + \
								   str(params['surprisal_cost'])
				model = SkipRNN(config_dict=params, emb_dict=embedding_dict, probs_dict=probs_dict)
				model.train()
			else:
				print(f"\n\nNetwork {csv_name} already exists\n\n")
