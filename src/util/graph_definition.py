"""
Graph creation functions.
"""


from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf

from rnn_cells.basic_rnn_cells import BasicLSTMCell, BasicGRUCell
from rnn_cells.skip_rnn_cells import SkipLSTMCell, MultiSkipLSTMCell
from rnn_cells.skip_rnn_cells import SkipGRUCell, MultiSkipGRUCell


def create_generic_flags():
    """
    Create flags which are shared by all experiments
    """
    # Generic flags
    tf.app.flags.DEFINE_string('model', 'lstm', "Select RNN cell: {lstm, gru, skip_lstm, skip_gru}")
    tf.app.flags.DEFINE_integer("rnn_cells", 64, "Number of RNN cells.")
    tf.app.flags.DEFINE_integer("rnn_layers", 1, "Number of RNN layers.")
    tf.app.flags.DEFINE_integer('batch_size', 32, "Batch size.")
    tf.app.flags.DEFINE_integer('epochs', 10, "Number of epochs")
    tf.app.flags.DEFINE_float('learning_rate', 0.01, "Learning rate.")
    tf.app.flags.DEFINE_float('grad_clip', 1., "Clip gradients at this value. Set to <=0 to disable clipping.")
    tf.app.flags.DEFINE_string('logdir', '../logs', "Directory where TensorBoard logs will be stored.")
    # tf.app.flags.DEFINE_string('embedding', 'simple', "Type of embedding used for the imdb task")

    # Flags for the Skip RNN cells
    tf.app.flags.DEFINE_float('cost_per_sample', 0.0001, "Cost per used sample. Set to 0 to disable this option.")
    tf.app.flags.DEFINE_float('surprisal_influence', 0.0001,
                              "How much it gets punished for average surprisal or discarded samples.")
    tf.app.flags.DEFINE_string('early_stopping', 'yes', "Whether to use early stopping or not")


def compute_gradients(loss, learning_rate, gradient_clipping=-1):
    """
    Create optimizer, compute gradients and (optionally) apply gradient clipping
    """
    opt = tf.train.AdamOptimizer(learning_rate)
    if gradient_clipping > 0:
        vars_to_optimize = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, vars_to_optimize), clip_norm=gradient_clipping)
        grads_and_vars = list(zip(grads, vars_to_optimize))
    else:
        grads_and_vars = opt.compute_gradients(loss)
    return opt, grads_and_vars


def create_model(model, num_cells, batch_size, learn_initial_state=True):
    """
    Returns a tuple of (cell, initial_state) to use with dynamic_rnn.
    If num_cells is an integer, a single RNN cell will be created. If it is a list, a stack of len(num_cells)
    cells will be created.
    """
    if not model in ['lstm', 'gru', 'skip_lstm', 'skip_gru']:
        raise ValueError('The specified model is not supported. Please use {lstm, gru, skip_lstm, skip_gru}.')
    if isinstance(num_cells, list) and len(num_cells) > 1:
        if model == 'skip_lstm':
            cells = MultiSkipLSTMCell(num_cells)
        elif model == 'skip_gru':
            cells = MultiSkipGRUCell(num_cells)
        elif model == 'lstm':
            cell_list = [BasicLSTMCell(n) for n in num_cells]
            cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        elif model == 'gru':
            cell_list = [BasicGRUCell(n) for n in num_cells]
            cells = tf.contrib.rnn.MultiRNNCell(cell_list)
        if learn_initial_state:
            if model == 'skip_lstm' or model == 'skip_gru':
                initial_state = cells.trainable_initial_state(batch_size)
            else:
                initial_state = []
                for idx, cell in enumerate(cell_list):
                    with tf.compat.v1.variable_scope('layer_%d' % (idx + 1)):
                        initial_state.append(cell.trainable_initial_state(batch_size))
                initial_state = tuple(initial_state)
        else:
            initial_state = None
        return cells, initial_state
    else:
        if isinstance(num_cells, list):
            num_cells = num_cells[0]
        if model == 'skip_lstm':
            cell = SkipLSTMCell(num_cells)
        elif model == 'skip_gru':
            cell = SkipGRUCell(num_cells)
        elif model == 'lstm':
            cell = BasicLSTMCell(num_cells)
        elif model == 'gru':
            cell = BasicGRUCell(num_cells)
        if learn_initial_state:
            initial_state = cell.trainable_initial_state(batch_size)
        else:
            initial_state = None
        return cell, initial_state


def using_skip_rnn(model):
    """
    Helper function determining whether a Skip RNN models is being used
    """
    return model.lower() == 'skip_lstm' or model.lower() == 'skip_gru'


def split_rnn_outputs(model, rnn_outputs):
    """
    Split the output of dynamic_rnn into the actual RNN outputs and the state update gate
    """
    if using_skip_rnn(model):
        return rnn_outputs.h, rnn_outputs.state_gate
    else:
        return rnn_outputs, tf.no_op()


def compute_budget_loss(model, loss, updated_states, cost_per_sample):
    """
    Compute penalization term on the number of updated states (i.e. used samples)
    """
    if using_skip_rnn(model):
        sample_loss = tf.reduce_mean(tf.reduce_sum(cost_per_sample * updated_states, 1), 0)
        # if not any(tf.is_nan(sample_loss)):
        #     return sample_loss
        # else:
        #     return tf.reduce_sum(cost_per_sample * updated_states, 1)
        # print = tf.cond(tf.math.is_nan(sample_loss),
        #                 true_fn=lambda: tf.Print(sample_loss, [sample_loss], "Sample loss is null "),
        #                 false_fn=lambda: tf.no_op())
        # with tf.control_dependencies([print]):
        return sample_loss
    else:
        return tf.zeros(loss.get_shape())

def compute_surprisal_loss(model, loss, updated_states, sample_probabilities, surprisal_influence):
    """
    Compute penalization term on the average surprisal of the unread samples.
    """
    if using_skip_rnn(model):
        neg_updated_states = tf.subtract(tf.ones(updated_states.get_shape(), dtype=tf.dtypes.float32), updated_states)
        surprisal_values = tf.multiply(tf.constant(-1.0), (tf.log(sample_probabilities)))
        # printer_0 = tf.Print(neg_updated_states, [neg_updated_states], "Inverse of the updated states is ")
        surprisals = tf.multiply(neg_updated_states,
                                 tf.where(tf.is_nan(surprisal_values), tf.zeros_like(surprisal_values),
                                          surprisal_values))
        tot_surprisal = tf.reduce_sum(surprisals)
        # printer_1 = tf.Print(tot_surprisal, [tot_surprisal], "Total surprisal is ")
        non_read_samples = tf.reduce_sum(neg_updated_states)
        # printer_2 = tf.Print(non_read_samples, [non_read_samples], "Non read samples is ")
        # with tf.control_dependencies([printer_0, printer_1, printer_2]):
        average_surprisal = tf.div_no_nan(tot_surprisal, non_read_samples)
        surprisal_loss = surprisal_influence * average_surprisal
        # print = tf.cond(tf.math.is_nan(surprisal_loss),
        #                 true_fn=lambda: tf.Print(surprisal_loss, [surprisal_loss], "Surprisal loss is null "),
        #                 false_fn=lambda: tf.no_op())
        # with tf.control_dependencies([print]):
        return surprisal_loss
    else:
        return tf.zeros(loss.get_shape())
