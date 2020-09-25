from sklearn.model_selection import ParameterGrid
from imdb import SkipRNN
from imdb import get_embedding_dicts
from imdb_no_surprisal import no_surp_SkipRNN
import tensorflow as tf
import argparse
import os
import gc
import time
from IPython.utils.io import Tee
from contextlib import closing
import GPUtil
from threading import Thread


class Monitor(Thread):
	def __init__(self, delay):
		super(Monitor, self).__init__()
		self.stopped = False
		self.delay = delay  # Time between calls to GPUtil
		self.start()

	def run(self):
		while not self.stopped:
			GPUtil.showUtilization()
			time.sleep(self.delay)

	def stop(self):
		self.stopped = True


# command_configs = {
# 	'learning_rate': [0.0001],
# 	'batch_size': [64],
# 	'hidden_units': [32],
# 	'cost_per_sample': [0.01, 0.001, 0.0001],
# 	'surprisal_cost': [0, 0.1, 0.01, 0.001]
# }
command_configs = {
	# 'learning_rate': [0.0005, 0.00075]
	'learning_rate': [0.00025],
	'batch_size': [64],
	'hidden_units': [32],
	'cost_per_sample': [1e-5],
	'surprisal_cost': [0, 0.1]  # 0 or whatever is best
}

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--id", type=int, help="id of the specific run")
	parser.add_argument("--tot_exps", type=int, default=2, help="The total amount of parallel experiments")
	parser.add_argument("--trials", type=int, default=1, help="The amount of times the same network is trained.")
	parser.add_argument("--print_gputil", type=bool, default=False,
						help="Whether to show the GPU utilization on terminal")
	parser.add_argument("--reverse")

	args = parser.parse_args()
	exp_id = args.id
	tot_exps = args.tot_exps
	n_trials = args.trials
	gputil = args.print_gputil

	gpus = tf.config.experimental.list_physical_devices('GPU')

	if not os.path.exists('../terminal_logs'):
		os.makedirs('../terminal_logs')

	with closing(Tee(f"../terminal_logs/exp{exp_id}.txt", "w", channel="stderr")) as outputstream:
		# with closing(Tee(f"../terminal_logs/exp{exp_id}_new_epochs.txt", "w", channel="stderr")) as outputstream:
		if gputil:
			monitor = Monitor(30)
		# with StdoutTee(f"../terminal_logs/exp{exp_id}.txt"), StderrTee(f"../terminal_logs/exp{exp_id}_err.txt"):
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
		grid = ParameterGrid(command_configs)
		n_nets = len(grid)
		for trial in range(0, n_trials):
			for idx, params in enumerate(grid):
				idx += trial * n_nets
				if idx % tot_exps == exp_id:
					file_name = 'EXP' + str(exp_id) + '_LR' + str(params['learning_rate']) + '_BS' + str(
						params['batch_size']) + '_HU' + str(params['hidden_units']) + '_CPS' + str(
						params['cost_per_sample']) + '_SC' + str(params['surprisal_cost']) + '_T' + str(trial)
					if not os.path.exists('../csvs/' + file_name + ".csv"):
						params['trial'] = trial
						params['epochs'] = 125
						params['early_stopping'] = 'yes'
						params['file_name'] = file_name
						if params['surprisal_cost'] == 0:
							model = no_surp_SkipRNN(config_dict=params, emb_dict=embedding_dict, probs_dict=probs_dict)
						else:
							model = SkipRNN(config_dict=params, emb_dict=embedding_dict, probs_dict=probs_dict)
						model.train()
						gc.collect()
		if gputil:
			monitor.close()
