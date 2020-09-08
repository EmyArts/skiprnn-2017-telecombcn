"""
Generic functions that are used in different scripts.
"""

from __future__ import absolute_import
from __future__ import print_function

import types
from decimal import Decimal
import numpy as np

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def print_setup(task_specific_setup=None):
    """
    Print experimental setup
    :param task_specific_setup: (optional) function printing task-specific parameters
    """
    model_dict = {'lstm': 'LSTM', 'gru': 'GRU', 'skip_lstm': 'SkipLSTM', 'skip_gru': 'SkipGRU'}
    print('\n\n\tExperimental setup')
    print('\t------------------\n')
    print('\tModel: %s' % model_dict[FLAGS.model.lower()])
    print('\tNumber of layers: %d' % FLAGS.rnn_layers)
    print('\tNumber of cells: %d' % FLAGS.rnn_cells)
    print('\tBatch size: %d' % FLAGS.batch_size)
    print('\tLearning rate: %.2E' % Decimal(FLAGS.learning_rate))


    if FLAGS.grad_clip > 0:
        print('\tGradient clipping: %.1f' % FLAGS.grad_clip)
    else:
        print('\tGradient clipping: No')

    if FLAGS.model.lower().startswith('skip'):
        print('\tCost per sample: %.2E' % Decimal(FLAGS.cost_per_sample))
        print('\tSurprisal cost: %.2E' % Decimal(FLAGS.surprisal_influence))

    if isinstance(task_specific_setup, types.FunctionType):
        print('')
        task_specific_setup()

    print('\n\n')


def compute_used_samples(update_state_gate):
    """
    Compute number of used samples (i.e. number of updated states)
    :param update_state_gate: values for the update state gate
    :return: number of used samples
    """
    batch_size = update_state_gate.shape[0]
    steps = 0.
    for idx in range(batch_size):
        for idt in range(update_state_gate.shape[1]):
            steps += update_state_gate[idx, idt]
    return steps / batch_size

def stats_used_samples(update_state_gate, embeddings, probs):
    try:
        assert (update_state_gate.shape == embeddings.shape[0, 1] and update_state_gate == probs.shape)
    except:
        print("Dimensions in stats_used_samples do not correspond")
        print(f"updates: {update_state_gate.shape}, embeddings: {embeddings.shape}, probs: {probs.shape}.")
        raise
    read_embs = []
    non_read_embs = []
    read_surps = []
    non_read_surps = []
    batch_size = update_state_gate.shape[0]
    for idx in range(batch_size):
        for idt in range(update_state_gate.shape[1]):
            if update_state_gate[idx, idt] == 0:
                non_read_embs.append(embeddings[idx, idt])
                non_read_surps.append(probs[idx, idt])
            else:
                read_embs.append(embeddings[idx, idt])
                read_surps.append(probs[idx, idt])
    read_surps = np.multiply(-1, np.log(read_surps))
    non_read_surps = np.multiply(-1, np.log(non_read_surps))
    return read_embs, non_read_embs, read_surps, non_read_surps


def scalar_summary(name, value):
    summary = tf.summary.Summary()
    summary_value = summary.value.add()
    summary_value.simple_value = value
    summary_value.tag = name
    return summary


