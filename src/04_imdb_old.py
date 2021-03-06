"""
Train RNN models on sequential MNIST, where inputs are processed pixel by pixel.

Results should be reported by evaluating on the test set the model with the best performance on the validation set.
To avoid storing checkpoints and having a separate evaluation script, this script evaluates on both validation and
test set after every epoch.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import time
import datetime

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow_datasets as tfds

import numpy as np
import matplotlib.pyplot as plt

from util.misc import *
from util.graph_definition import *
from util.embedder import Embedding
from util.gensimw2v import Gensim_Embedding
# Task-independent flags
create_generic_flags()

# Task-specific flags
tf.app.flags.DEFINE_string('data_path', '../data', "Path where the MNIST data will be stored.")
FLAGS = tf.app.flags.FLAGS

# Constants
OUTPUT_SIZE = 2
SEQUENCE_LENGTH = 2520
VALIDATION_SAMPLES = 500 # Just for debugging
#VALIDATION_SAMPLES = 5000
NUM_EPOCHS = 12

# Load data
imdb_builder = tfds.builder('imdb_reviews/plain_text', data_dir=FLAGS.data_path)
imdb_builder.download_and_prepare()
info = imdb_builder.info
if FLAGS.embedding == 'simple':
    embedder = Embedding()
else:
    embedder = Gensim_Embedding()

SEQUENCE_LENGTH = embedder.max_sent_len
EMB_TABLE = embedder.embedding_matrix()
# EMBEDDING_LENGTH =
# datasets = mnist_builder.as_dataset()

#Originalli 25k for training and 25k for testing -> 20k for testing and 5k for validation
TRAIN_SAMPLES = info.splits[tfds.Split.TRAIN].num_examples
TEST_SAMPLES = info.splits[tfds.Split.TEST].num_examples - VALIDATION_SAMPLES

ITERATIONS_PER_EPOCH = int(TRAIN_SAMPLES / FLAGS.batch_size)
VAL_ITERS = int(VALIDATION_SAMPLES / FLAGS.batch_size)
TEST_ITERS = int(TEST_SAMPLES / FLAGS.batch_size)


def input_fn(split):
    # Reset datasets once done debugging
    test_split = f'test[:500]'
    valid_split = f'test[500:750]'
    # test_split = f'test[:{TEST_SAMPLES}]'
    # valid_split = f'test[{TEST_SAMPLES}:]'
    if split == 'train':
        dataset = imdb_builder.as_dataset(as_supervised=True, split='train[:5000]')
        size = 5000
        #print("Total amount of training samples: " + str(len(list(dataset))))
    elif split == 'val':
        dataset = imdb_builder.as_dataset(as_supervised=True, split=valid_split)
        size = 250
        #print("Total amount of validation samples: " + str(len(list(dataset))))
    elif split == 'test':
        dataset = imdb_builder.as_dataset(as_supervised=True, split=test_split)
        size = 500
        #print("Total amount of test samples: " + str(len(list(dataset))))
    else:
        raise ValueError()

    try:
        inputs, ps, l = embedder.get_embeddings(dataset)
        # dataset = embedder.tay_get_embedding(dataset, size)
        print(dataset)
    except Exception as e:
        print("An exception occured during the get embeddings.")
        print(e)


    # dataset = dataset.repeat()
    # dataset = dataset.batch(FLAGS.batch_size)
    # dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # iterator = dataset.make_initializable_iterator()
    # text, probs, labels = iterator.get_next()
    # iterator_init_op = iterator.initializer
    inputs = {'text': inputs, 'probs': ps, 'labels': l}
    # inputs = {'text': text, 'labels': labels, 'iterator_init_op': iterator_init_op}
    #print(f"\n\n Input shape is {text.shape}, probs shape is {probs.shape}, labels shape is {labels.shape}")
    return inputs


def model_fn(mode, inputs, reuse=False):
    #emb_table = tf.convert_to_tensor(embedder.embedding_matrix())
    samples = tf.gather(EMB_TABLE, inputs["text"], axis=0, batch_dims=1)

    #samples = samples.reshape(samples, (-1, SEQUENCE_LENGTH, embedder.vector_length()))
    # probs = tf.reshape(inputs["probs"], (-1, SEQUENCE_LENGTH, 1))
    # samples = inputs["text"]

    print_samples = tf.Print(samples, [samples], "\nSamples are: \n")
    ground_truth = tf.cast(inputs['labels'], tf.int64)

    is_training = (mode == 'train')

    with tf.variable_scope('model', reuse=reuse):
        cell, initial_state = create_model(model=FLAGS.model,
                                           num_cells=[FLAGS.rnn_cells] * FLAGS.rnn_layers,
                                           batch_size=FLAGS.batch_size)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, samples, dtype=tf.float32, initial_state=initial_state)

        # Split the outputs of the RNN into the actual outputs and the state update gate
        rnn_outputs, updated_states = split_rnn_outputs(FLAGS.model, rnn_outputs)

        print(f"\nUpdated states are {updated_states}.\n")

        logits = layers.linear(inputs=rnn_outputs[:, -1, :], num_outputs=OUTPUT_SIZE)
        predictions = tf.argmax(logits, 1)

    # Compute cross-entropy loss
    cross_entropy_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth)
    cross_entropy = tf.reduce_mean(cross_entropy_per_sample)

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), tf.float32))

    # Compute loss for each updated state
    budget_loss = compute_budget_loss(FLAGS.model, cross_entropy, updated_states, FLAGS.cost_per_sample)
    #print(f"Budget loss: {tf.make_ndarray(budget_loss)}")

    # surprisal_loss = compute_surprisal_loss(FLAGS.model, cross_entropy, updated_states, probs, 0.0001)
    #print(f"Surprisal loss: {tf.make_ndarray(surprisal_loss)}")

    # Combine all losses
    loss = cross_entropy + budget_loss# + surprisal_loss
    loss = tf.reshape(loss, [])

    if is_training:
        # Optimizer
        opt, grads_and_vars = compute_gradients(loss, FLAGS.learning_rate, FLAGS.grad_clip)
        train_fn = opt.apply_gradients(grads_and_vars)

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['samples'] = print_samples
    model_spec['labels'] = ground_truth
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['updated_states'] = updated_states

    if is_training:
        model_spec['train_fn'] = train_fn

    return model_spec


def train():
    train_inputs = input_fn(split='train')
    valid_inputs = input_fn(split='val')
    test_inputs = input_fn(split='test')

    train_model_spec = model_fn('train', train_inputs)
    valid_model_spec = model_fn('val', valid_inputs, reuse=True)
    test_model_spec = model_fn('test', test_inputs, reuse=True)

    sess = tf.Session()

    log_dir = os.path.join(FLAGS.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    valid_writer = tf.summary.FileWriter(log_dir + '/val')
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Initialize weights
    sess.run(train_model_spec['variable_init_op'])

    try:
        train_acc_plt = np.empty((NUM_EPOCHS, ITERATIONS_PER_EPOCH))
        val_acc_plt = np.empty((NUM_EPOCHS, VAL_ITERS))

        for epoch in range(NUM_EPOCHS):
            train_fn = train_model_spec['train_fn']

            # Load the training dataset into the pipeline
            # sess.run(train_model_spec['iterator_init_op'])
            sess.run(train_model_spec)
            # sess.run(train_model_spec['samples'])

            start_time = time.time()
            for iteration in range(ITERATIONS_PER_EPOCH):
                # Perform SGD update
                sess.run([train_fn])
                loss = sess.run(train_model_spec['loss'])
                # sess.run(train_model_spec['samples'])
                print(loss)
                train_acc_plt[epoch][iteration] = loss
            duration = time.time() - start_time

            # Evaluate on validation data
            accuracy = valid_model_spec['accuracy']
            loss = valid_model_spec['loss']
            updated_states = valid_model_spec['updated_states']
            #samples = tf.make_ndarray(tf.make_tensor_proto(valid_model_spec['samples']))

            # Load the validation dataset into the pipeline
            # sess.run(valid_model_spec['iterator_init_op'])
            sess.run(valid_model_spec)

            valid_accuracy, valid_loss, valid_steps = 0, 0, 0
            for iter in range(VAL_ITERS):
                valid_iter_accuracy, valid_iter_loss, valid_used_inputs = sess.run([accuracy, loss, updated_states])
                valid_loss += valid_iter_loss
                valid_accuracy += valid_iter_accuracy
                if valid_used_inputs is not None:
                    valid_steps += compute_used_samples(valid_used_inputs)
                else:
                    valid_steps += SEQUENCE_LENGTH
                val_acc_plt[epoch][iter] = valid_iter_loss
            valid_accuracy /= VAL_ITERS
            valid_loss /= VAL_ITERS
            valid_steps /= VAL_ITERS

            valid_writer.add_summary(scalar_summary('accuracy', valid_accuracy), epoch)
            #valid_writer.add_summary(scalar_summary('samples', samples), epoch)
            valid_writer.add_summary(scalar_summary('loss', valid_loss), epoch)
            valid_writer.add_summary(scalar_summary('used_samples', valid_steps / SEQUENCE_LENGTH), epoch)
            valid_writer.flush()

            # Evaluate on test data
            accuracy = test_model_spec['accuracy']
            loss = test_model_spec['loss']
            updated_states = test_model_spec['updated_states']

            # Load the test dataset into the pipeline
            # sess.run(test_model_spec['iterator_init_op'])
            sess.run(test_model_spec)

            test_accuracy, test_loss, test_steps = 0, 0, 0
            for _ in range(TEST_ITERS):
                test_iter_accuracy, test_iter_loss, test_used_inputs = sess.run([accuracy, loss, updated_states])
                test_accuracy += test_iter_accuracy
                if test_used_inputs is not None:
                    test_steps += compute_used_samples(test_used_inputs)
                else:
                    test_steps += SEQUENCE_LENGTH
            test_accuracy /= TEST_ITERS
            test_loss /= TEST_ITERS
            test_steps /= TEST_ITERS

            test_writer.add_summary(scalar_summary('accuracy', test_accuracy), epoch)
            test_writer.add_summary(scalar_summary('loss', test_loss), epoch)
            test_writer.add_summary(scalar_summary('used_samples', test_steps / SEQUENCE_LENGTH), epoch)
            test_writer.flush()

            print("Epoch %d/%d, "
                  "duration: %.2f seconds, " 
                  "validation accuracy: %.2f%%, "
                  "validation samples: %.2f (%.2f%%), "
                  "test accuracy: %.2f%%, "
                  "test samples: %.2f (%.2f%%)" % (epoch + 1,
                                                   NUM_EPOCHS,
                                                   duration,
                                                   100. * valid_accuracy,
                                                   valid_steps,
                                                   100. * valid_steps / SEQUENCE_LENGTH,
                                                   100. * test_accuracy,
                                                   test_steps,
                                                   100. * test_steps / SEQUENCE_LENGTH))

        # Training curve for epochs
        plt.plot(train_acc_plt[:, 0], label='Training loss')
        plt.plot(val_acc_plt[:, 0], label='Validation loss')
        plt.title("Training curve for epochs")
        plt.savefig("Epoch.png")
        plt.show()

        plt.plot(train_acc_plt.flatten(), label='Training loss')
        plt.title("Training curve for all iterations")
        plt.savefig("train_iter.png")
        plt.show()

        plt.plot(val_acc_plt.flatten(), label='Validation loss')
        plt.title("Validation curve for all iterations")
        plt.savefig("val_iter.png")
        plt.show()

    except KeyboardInterrupt:
        pass


def main(argv=None):
    print_setup()
    train()


if __name__ == '__main__':
    tf.app.run()
