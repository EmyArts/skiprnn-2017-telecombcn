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
import pickle

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow_datasets as tfds

import numpy as np
import math
import matplotlib.pyplot as plt

from util.misc import *
from util.graph_definition import *
from gensim.utils import tokenize
# Task-independent flags
create_generic_flags()

# Task-specific flags
tf.app.flags.DEFINE_string('data_path', '../data', "Path where the MNIST data will be stored.")
FLAGS = tf.app.flags.FLAGS

# Constants
OUTPUT_SIZE = 2
SEQUENCE_LENGTH = 2520
#VALIDATION_SAMPLES = 500 # Just for debugging
#VALIDATION_SAMPLES = 5000
EMBEDDING_LENGTH = 50
NUM_EPOCHS = FLAGS.epochs
BATCH_SIZE = FLAGS.batch_size

PROBS_FILE = 'probs.pkl'

# Load data
imdb_builder = tfds.builder('imdb_reviews/plain_text', data_dir=FLAGS.data_path)
imdb_builder.download_and_prepare()
info = imdb_builder.info

# datasets = mnist_builder.as_dataset()

#Originalli 25k for training and 25k for testing -> 15k for validation and 10k for testing
TRAIN_SAMPLES = 10000
TEST_SAMPLES = 5000
# TRAIN_SAMPLES = info.splits[tfds.Split.TRAIN].num_examples
# TEST_SAMPLES = info.splits[tfds.Split.TEST].num_examples

ITERATIONS_PER_EPOCH = int(TRAIN_SAMPLES / FLAGS.batch_size)
TEST_ITERS = int(TEST_SAMPLES / FLAGS.batch_size)
# ITERATIONS_PER_EPOCH = int(5000/BATCH_SIZE)
# TEST_ITERS = int(5000/BATCH_SIZE)

PROBS_DICT = pickle.load(open(PROBS_FILE, 'rb'))

EMBEDDING_DICT = {}
f = open(f'glove.6B.{str(EMBEDDING_LENGTH)}d.txt')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    EMBEDDING_DICT[word] = coefs
f.close()
print('Total %s word vectors.' % len(EMBEDDING_DICT))

def input_fn(split):
    # Reset datasets once done debugging
    test_split = f'test[:{TEST_SAMPLES}]'
    # test_split = f'test[:{TEST_SAMPLES}]'
    # valid_split = f'test[{TEST_SAMPLES}:]'
    if split == 'train':
        data = imdb_builder.as_dataset(as_supervised=True, split=f'train[:{TRAIN_SAMPLES}]')
        tot_len = math.ceil(TRAIN_SAMPLES/BATCH_SIZE)  #This will be the ITERATIONS_PER_EPOCH
        #print("Total amount of training samples: " + str(len(list(dataset))))
        #print("Total amount of validation samples: " + str(len(list(dataset))))
    elif split == 'test':
        data = imdb_builder.as_dataset(as_supervised=True, split=test_split)
        tot_len = math.ceil(TEST_SAMPLES/BATCH_SIZE) # This will be TEST_ITERS
        #print("Total amount of test samples: " + str(len(list(dataset))))
    else:
        raise ValueError()

    # print(f"Vector for unknonw words is {embeddings_index.get('unk')}")
    embedding_matrix = np.zeros((tot_len, BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_LENGTH), dtype=np.float32)
    probs_matrix = np.ones((tot_len, BATCH_SIZE, SEQUENCE_LENGTH, 1), dtype=np.float32)
    labels = np.empty((tot_len, BATCH_SIZE), dtype=np.int64)
    line_index = 0
    batch_index = 0
    c_unk = 0
    word_count = 0
    entry = 0
    for text, label in tfds.as_numpy(data):
        labels[batch_index][entry] = label
        # print(label)
        tokens = list(tokenize(str(text), lowercase=True))[3:]
        for i, t in enumerate(tokens):
            embedding_vector = EMBEDDING_DICT.get(t)
            prob = PROBS_DICT.get(t)
            if embedding_vector is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[batch_index][entry][i] = embedding_vector
                probs_matrix[batch_index][entry][i] = prob
            else:
                c_unk += 1
            word_count += 1
        line_index += 1
        entry = line_index % BATCH_SIZE
        if entry == 0:
            batch_index += 1
    print(f"{c_unk} words out of {word_count} total words unknown for {split} set.")

    # inputs = {'text': text, 'labels': labels, 'iterator_init_op': iterator_init_op}
    print(f"\n\n Input shape is {embedding_matrix.shape},  labels shape is {labels.shape}, probs shape is {probs_matrix.shape}")
    # np.expand_dims(probs_matrix, axis=-1)
    return embedding_matrix, labels, probs_matrix

# print_samples = tf.Print(samples, [samples], "\nSamples are: \n")

def train():
    samples = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH, EMBEDDING_LENGTH], name='Samples')  # (batch, time, in)
    ground_truth = tf.placeholder(tf.int64, shape=[BATCH_SIZE], name='GroundTruth')
    probs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQUENCE_LENGTH, 1], name='Probs')

    cell, initial_state = create_model(model=FLAGS.model,
                                       num_cells=[FLAGS.rnn_cells] * FLAGS.rnn_layers,
                                       batch_size=FLAGS.batch_size)

    rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, samples, dtype=tf.float32, initial_state=initial_state)

    # Split the outputs of the RNN into the actual outputs and the state update gate
    rnn_outputs, updated_states = split_rnn_outputs(FLAGS.model, rnn_outputs)

    # print(f"\nUpdated states are {updated_states}.\n")

    logits = layers.linear(inputs=rnn_outputs[:, -1, :], num_outputs=OUTPUT_SIZE)
    predictions = tf.argmax(logits, 1)

    # Compute cross-entropy loss
    cross_entropy_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth)
    cross_entropy = tf.reduce_mean(cross_entropy_per_sample)


    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), tf.float32))

    # Compute loss for each updated state
    budget_loss = compute_budget_loss(FLAGS.model, cross_entropy, updated_states, FLAGS.cost_per_sample)

    # Compute loss for the amount of surprisal
    surprisal_loss = compute_surprisal_loss(FLAGS.model, cross_entropy, updated_states, probs, FLAGS.surprisal_influence)
    # Avoid encouraging to not skip.
    surprisal_loss = tf.where(tf.equal(surprisal_loss, tf.zeros_like(surprisal_loss)), tf.ones_like(surprisal_loss), surprisal_loss)

    loss = cross_entropy + budget_loss + surprisal_loss
    loss = tf.reshape(loss, [])

    loss = tf.where(tf.is_nan(loss), tf.ones_like(loss), loss)


    # Optimizer
    opt, grads_and_vars = compute_gradients(loss, FLAGS.learning_rate, FLAGS.grad_clip)
    train_fn = opt.apply_gradients(grads_and_vars)

    sess = tf.Session()

    log_dir = os.path.join(FLAGS.logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    # Initialize weights
    sess.run(tf.global_variables_initializer())

    try:
        train_matrix, train_labels, train_probs = input_fn(split='train')
        test_matrix, test_labels, test_probs = input_fn(split='test')

        train_acc_plt = np.empty((NUM_EPOCHS, ITERATIONS_PER_EPOCH))
        val_acc_plt = np.empty((NUM_EPOCHS))
        loss_plt = np.empty((NUM_EPOCHS, ITERATIONS_PER_EPOCH, 3))

        for epoch in range(NUM_EPOCHS):

            # Load the training dataset into the pipeline
            # sess.run(train_model_spec['iterator_init_op'])

            # sess.run(train_model_spec['samples'])

            start_time = time.time()
            train_accuracy, train_steps, train_loss = 0, 0, 0
            for iteration in range(ITERATIONS_PER_EPOCH):
                # Perform SGD update
                # print(iteration, train_probs[iteration].shape)
                out = sess.run([train_fn, loss, accuracy, updated_states, cross_entropy, budget_loss, surprisal_loss],
                               feed_dict={samples: train_matrix[iteration],
                                          ground_truth: train_labels[iteration],
                                          probs: train_probs[iteration]
                                          })
                train_accuracy += out[2]
                train_loss += out[1]
                loss_plt[epoch][iteration] = out[4:] # entropy, budget, surprisal
                if out[3] is not None:
                    train_steps += compute_used_samples(out[3])
                else:
                    train_steps += SEQUENCE_LENGTH
            duration = time.time() - start_time

            train_accuracy /= ITERATIONS_PER_EPOCH
            train_loss /= ITERATIONS_PER_EPOCH
            train_steps /= ITERATIONS_PER_EPOCH


            test_accuracy, test_loss, test_steps = 0, 0, 0
            for iteration in range(TEST_ITERS):
                test_iter_accuracy, test_iter_loss, test_used_inputs = sess.run([accuracy, loss, updated_states],
                                                                                feed_dict={samples: test_matrix[iteration],
                                                                                           ground_truth: test_labels[iteration],
                                                                                           probs: test_probs[iteration]
                                                                                           })
                test_accuracy += test_iter_accuracy
                test_loss += test_iter_loss
                if test_used_inputs is not None:
                    test_steps += compute_used_samples(test_used_inputs)
                else:
                    test_steps += SEQUENCE_LENGTH
            test_accuracy /= TEST_ITERS
            test_loss /= TEST_ITERS
            test_steps /= TEST_ITERS
            val_acc_plt[epoch] = test_loss

            test_writer.add_summary(scalar_summary('accuracy', test_accuracy), epoch)
            test_writer.add_summary(scalar_summary('loss', test_loss), epoch)
            test_writer.add_summary(scalar_summary('used_samples', test_steps / SEQUENCE_LENGTH), epoch)
            test_writer.flush()

            print("Epoch %d/%d, "
                  "duration: %.2f seconds, " 
                  "train accuracy: %.2f%%, "
                  "train samples: %.2f (%.2f%%)"
                  "test accuracy: %.2f%%, "
                  "test samples: %.2f (%.2f%%)" % (epoch + 1,
                                                   NUM_EPOCHS,
                                                   duration,
                                                   100. * train_accuracy,
                                                   train_steps,
                                                   100. * train_steps / SEQUENCE_LENGTH,
                                                   100. * test_accuracy,
                                                   test_steps,
                                                   100. * test_steps / SEQUENCE_LENGTH))

            loss_perc = loss_plt[epoch, :, :].mean()
            loss_perc = (loss_perc / loss_perc.sum()) * 100
            print(f"entropy: {loss_perc[0]}%, budget: {loss_perc[1]}%, surprisal: {loss_perc[2]}%.")

            # print(f"entropy: {loss_plt[epoch, :, 0].mean()}, budget: {loss_plt[epoch, :, 1].mean()}, surprisal: {loss_plt[epoch, :, 2].mean()}.")

        # Training curve for epochs
        plt.plot(train_acc_plt[:, 0], label='Training loss')
        plt.plot(val_acc_plt, label='Test loss')
        plt.title("Training curve for epochs")
        plt.savefig("Epoch.png")
        plt.show()

        plt.plot(train_acc_plt.flatten(), label='Training loss')
        plt.title("Training curve for all iterations")
        plt.savefig("train_iter.png")
        plt.show()

        plt.plot(loss_plt[:, :, 0].flatten(), label='Entropy loss')
        plt.plot(loss_plt[:, :, 1].flatten(), label='Budget loss')
        plt.plot(loss_plt[:, :, 2].flatten(), label='Surprisal loss')
        plt.title("Training curve for all iterations")
        plt.savefig("train_iter.png")
        plt.show()

    except KeyboardInterrupt:
        pass


def main(argv=None):
    print_setup()
    train()


if __name__ == '__main__':
    tf.app.run()
