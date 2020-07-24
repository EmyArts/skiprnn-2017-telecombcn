from sklearn.model_selection import ParameterGrid
from imdb import SkipRNN
from imdb import get_embedding_dicts
import tensorflow as tf
import argparse
import wandb as wb

command_configs = {
	# 'learning_rate': [0.01, 0.001, 0.0001],
	'batch_size': [32, 64],
	'hidden_units': [32, 64, 96],
	'cost_per_sample': [0.005, 0.001, 0.0005, 0.0001],
	'surprisal_cost': [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
}

if __name__ == '__main__':

	wb.init()
	parser = argparse.ArgumentParser()
	parser.add_argument("--learning_rate", type=float, help="the learning rate")
	args = parser.parse_args()

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
	for params in list(ParameterGrid(command_configs)):
		params['learning_rate'] = args.learning_rate
		params['epochs'] = 50
		params['folder'] = '../LR' + str(params['learning_rate']) + '_BS' + str(params['batch_size']) + \
						   '_HU' + str(params['hidden_units']) + '_CPS' + str(params['cost_per_sample']) + '_SC' + \
						   str(params['surprisal_cost'])
		model = SkipRNN(config_dict=params, emb_dict=embedding_dict, probs_dict=probs_dict)
		model.train()
