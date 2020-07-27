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
import logging

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow_datasets as tfds

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

from util.misc import *
from util.graph_definition import *
from gensim.utils import tokenize


# Task-independent flags

class SkipRNN():

    def __init__(self, config_dict, emb_dict, probs_dict):

        self.EMBEDDING_DICT = emb_dict
        self.PROBS_DICT = probs_dict
        self.CONFIG_DICT = config_dict

        self.LEARNING_RATE = config_dict['learning_rate']
        self.NUM_EPOCHS = config_dict['epochs']
        self.BATCH_SIZE = config_dict['batch_size']
        self.HIDDEN_UNITS = config_dict['hidden_units']
        self.SURPRISAL_COST = config_dict['surprisal_cost']
        self.COST_PER_SAMPLE = config_dict['cost_per_sample']
        self.FOLDER = config_dict['folder']

        # Constants
        self.OUTPUT_SIZE = 2
        self.SEQUENCE_LENGTH = 2520
        self.EARLY_STOPPING = False
        # VALIDATION_SAMPLES = 5000
        self.EMBEDDING_LENGTH = 50

        # Originalli 25k for training and 25k for testing -> 15k for validation and 10k for testing
        # Keras used 15k for training, 10k for validation out of the training set and 25k for testing later
        self.TRAIN_SAMPLES = 12000
        self.VAL_SAMPLES = 8000
        # self.TRAIN_SAMPLES = 15000
        # self.VAL_SAMPLES = 10000
        # TRAIN and VAL samples should always sum up to 25k

        # TRAIN_SAMPLES = info.splits[tfds.Split.TRAIN].num_examples
        # TEST_SAMPLES = info.splits[tfds.Split.TEST].num_examples

        self.ITERATIONS_PER_EPOCH = int(self.TRAIN_SAMPLES / self.BATCH_SIZE)
        self.VAL_ITERS = int(self.VAL_SAMPLES / self.BATCH_SIZE)
        # ITERATIONS_PER_EPOCH = int(5000/BATCH_SIZE)
        # TEST_ITERS = int(5000/BATCH_SIZE)

        # Load data
        self.imdb_builder = tfds.builder('imdb_reviews/plain_text', data_dir='../data')
        self.imdb_builder.download_and_prepare()
        # info = imdb_builder.info

        # Setting up logger
        # os.mkdir(self.FOLDER)
        # logging.basicConfig(filename=f"{self.FOLDER}/log.log", filemode='w', format='%(asctime)s - %(message)s',
        #             level=logging.INFO)
        if not os.path.exists(self.FOLDER):
            os.makedirs(self.FOLDER)
        self.logger = logging.getLogger("Net Logger")
        fh = logging.FileHandler(f"{self.FOLDER}/log.log")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(fh)
        # self.logger.basicConfig(filename=f"{self.FOLDER}/log.log", filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)


    def input_fn(self, split):
        # Reset datasets once done debugging
        val_split = f'train[{self.TRAIN_SAMPLES}:{self.TRAIN_SAMPLES + self.VAL_SAMPLES}]'
        # test_split = f'test[{TRAIN_SAMPLES:{TEST_SAMPLES}]'
        # valid_split = f'test[{TEST_SAMPLES}:]'
        if split == 'train':
            data = self.imdb_builder.as_dataset(as_supervised=True, split=f'train[:{self.TRAIN_SAMPLES}]')
            tot_len = math.ceil(self.TRAIN_SAMPLES/self.BATCH_SIZE)  #This will be the ITERATIONS_PER_EPOCH
            #print("Total amount of training samples: " + str(len(list(dataset))))
            #print("Total amount of validation samples: " + str(len(list(dataset))))
        elif split == 'val':
            data = self.imdb_builder.as_dataset(as_supervised=True, split=val_split)
            tot_len = math.ceil(self.VAL_SAMPLES / self.BATCH_SIZE) # This will be VAL_ITERS
            #print("Total amount of test samples: " + str(len(list(dataset))))
        else:
            raise ValueError()

        # print(f"Vector for unknonw words is {embeddings_index.get('unk')}")
        embedding_matrix = np.zeros((tot_len, self.BATCH_SIZE, self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH), dtype=np.float32)
        probs_matrix = np.ones((tot_len, self.BATCH_SIZE, self.SEQUENCE_LENGTH, 1), dtype=np.float32)
        labels = np.empty((tot_len, self.BATCH_SIZE), dtype=np.int64)
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
                embedding_vector = self.EMBEDDING_DICT.get(t)
                prob = self.PROBS_DICT.get(t)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[batch_index][entry][i] = embedding_vector
                    probs_matrix[batch_index][entry][i] = prob
                else:
                    c_unk += 1
                word_count += 1
            line_index += 1
            entry = line_index % self.BATCH_SIZE
            if entry == 0:
                batch_index += 1
        self.logger.info(f"{c_unk} words out of {word_count} total words unknown for {split} set.")
        print(f"{c_unk} words out of {word_count} total words unknown for {split} set.")

        # inputs = {'text': text, 'labels': labels, 'iterator_init_op': iterator_init_op}
        print(f"\n\n Input shape is {embedding_matrix.shape},  labels shape is {labels.shape}, probs shape is {probs_matrix.shape}")
        # np.expand_dims(probs_matrix, axis=-1)
        return embedding_matrix, labels, probs_matrix

    # print_samples = tf.Print(samples, [samples], "\nSamples are: \n")

    # def train(hyper_params = BASE_CONF):
    def train(self):
        samples = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH], name='Samples')  # (batch, time, in)
        ground_truth = tf.placeholder(tf.int64, shape=[self.BATCH_SIZE], name='GroundTruth')
        probs = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH, 1], name='Probs')

        cell, initial_state = create_model(model='skip_lstm',
                                           num_cells=[self.HIDDEN_UNITS],
                                           batch_size=self.BATCH_SIZE)

        rnn_outputs, rnn_states = tf.nn.dynamic_rnn(cell, samples, dtype=tf.float32, initial_state=initial_state)

        # Split the outputs of the RNN into the actual outputs and the state update gate
        rnn_outputs, updated_states = split_rnn_outputs('skip_lstm', rnn_outputs)

        # print(f"\nUpdated states are {updated_states}.\n")

        logits = layers.linear(inputs=rnn_outputs[:, -1, :], num_outputs=self.OUTPUT_SIZE)
        predictions = tf.argmax(logits, 1)

        # Compute cross-entropy loss
        cross_entropy_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ground_truth)
        cross_entropy = tf.reduce_mean(
            tf.where(tf.math.is_nan(cross_entropy_per_sample), tf.ones(cross_entropy_per_sample.get_shape()),
                     cross_entropy_per_sample))


        # Compute accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), tf.float32))

        # Compute loss for each updated state
        budget_loss = compute_budget_loss('skip_lstm', cross_entropy, updated_states, self.COST_PER_SAMPLE)

        # Compute loss for the amount of surprisal
        surprisal_loss = compute_surprisal_loss('skip_lstm', cross_entropy, updated_states, probs, self.SURPRISAL_COST)
        # Avoid encouraging to not skip.
        surprisal_loss = tf.where(tf.equal(surprisal_loss, tf.zeros_like(surprisal_loss)), tf.ones_like(surprisal_loss), surprisal_loss)

        loss = cross_entropy + budget_loss + surprisal_loss
        loss = tf.reshape(loss, [])

        loss = tf.where(tf.is_nan(loss), tf.ones_like(loss), loss)


        # Optimizer
        opt, grads_and_vars = compute_gradients(loss, self.LEARNING_RATE, 1) #1 is for gradient clipping
        train_fn = opt.apply_gradients(grads_and_vars)

        sess = tf.Session()

        # log_dir = os.path.join(self.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # val_writer = tf.summary.FileWriter(log_dir + '/validation')

        # Initialize weights
        sess.run(tf.global_variables_initializer())

        train_loss_plt = np.empty((self.NUM_EPOCHS))
        val_loss_plt = np.empty((self.NUM_EPOCHS))
        loss_plt = np.empty((self.NUM_EPOCHS, self.ITERATIONS_PER_EPOCH, 3))
        val_acc_df = np.empty((self.NUM_EPOCHS))
        train_acc_df = np.empty((self.NUM_EPOCHS))
        train_update_df = np.empty((self.NUM_EPOCHS))
        val_update_df = np.empty((self.NUM_EPOCHS))

        FILE_NAME = f'hu{self.HIDDEN_UNITS}_bs{self.BATCH_SIZE}_lr{self.LEARNING_RATE}_b{self.COST_PER_SAMPLE}_s{self.SURPRISAL_COST}'

        try:
            train_matrix, train_labels, train_probs = self.input_fn(split='train')
            val_matrix, val_labels, val_probs = self.input_fn(split='val')

            # train_loss_plt = np.empty((self.NUM_EPOCHS, self.ITERATIONS_PER_EPOCH)

            for epoch in range(self.NUM_EPOCHS):

                # Load the training dataset into the pipeline
                # sess.run(train_model_spec['iterator_init_op'])

                # sess.run(train_model_spec['samples'])

                start_time = time.time()
                train_accuracy, train_steps, train_loss = 0, 0, 0
                for iteration in range(self.ITERATIONS_PER_EPOCH):
                    # Perform SGD update
                    # print(iteration, train_probs[iteration].shape)
                    out = sess.run([train_fn, loss, accuracy, updated_states, cross_entropy, budget_loss, surprisal_loss],
                                   feed_dict={samples: train_matrix[iteration],
                                              ground_truth: train_labels[iteration],
                                              probs: train_probs[iteration]
                                              })
                    train_accuracy += out[2]
                    train_loss += out[1]
                    loss_plt[epoch][iteration] = out[4:]  # entropy, budget, surprisal
                    if out[3] is not None:
                        train_steps += compute_used_samples(out[3])
                    else:
                        train_steps += self.SEQUENCE_LENGTH
                duration = time.time() - start_time

                train_accuracy /= self.ITERATIONS_PER_EPOCH
                train_loss /= self.ITERATIONS_PER_EPOCH
                train_steps /= self.ITERATIONS_PER_EPOCH
                train_loss_plt[epoch] = train_loss
                train_acc_df[epoch] = train_accuracy
                train_update_df[epoch] = train_steps

                val_accuracy, val_loss, val_steps = 0, 0, 0
                for iteration in range(self.VAL_ITERS):
                    val_iter_accuracy, val_iter_loss, val_used_inputs = sess.run([accuracy, loss, updated_states],
                                                                                 feed_dict={
                                                                                     samples: val_matrix[iteration],
                                                                                     ground_truth: val_labels[
                                                                                         iteration],
                                                                                     probs: val_probs[iteration]
                                                                                     })
                    val_accuracy += val_iter_accuracy
                    val_loss += val_iter_loss
                    if val_used_inputs is not None:
                        val_steps += compute_used_samples(val_used_inputs)
                    else:
                        val_steps += self.SEQUENCE_LENGTH
                val_accuracy /= self.VAL_ITERS
                val_loss /= self.VAL_ITERS
                val_steps /= self.VAL_ITERS
                val_loss_plt[epoch] = val_loss
                val_acc_df[epoch] = val_accuracy
                val_update_df[epoch] = val_steps

                # val_writer.add_summary(scalar_summary('accuracy', val_accuracy), epoch)
                # val_writer.add_summary(scalar_summary('loss', val_loss), epoch)
                # val_writer.add_summary(scalar_summary('used_samples', val_steps / self.SEQUENCE_LENGTH), epoch)
                # val_writer.flush()

                print("Epoch %d/%d, "
                      "duration: %.2f seconds, "
                      "train accuracy: %.2f%%, "
                      "train samples: %.2f (%.2f%%), "
                      "val accuracy: %.2f%%, "
                      "val samples: %.2f (%.2f%%)" % (epoch + 1,
                                                       self.NUM_EPOCHS,
                                                       duration,
                                                       100. * train_accuracy,
                                                       train_steps,
                                                       100. * train_steps / self.SEQUENCE_LENGTH,
                                                       100. * val_accuracy,
                                                       val_steps,
                                                       100. * val_steps / self.SEQUENCE_LENGTH))

                loss_abs = loss_plt[epoch].mean(axis=0)
                print("Absolute losses: entropy: %.3f, budget: %.3f, surprisal: %.3f." % (loss_abs[0], loss_abs[1], loss_abs[2]))
                loss_perc = np.divide(loss_abs, (loss_abs.sum())) * 100
                print("Percentage losses: entropy: %.2f%%, budget: %.2f%%, surprisal: %.2f%%.\n" % (loss_perc[0], loss_perc[1], loss_perc[2]))

                self.logger.info("Epoch %d/%d, "
                      "duration: %.2f seconds, "
                      "train accuracy: %.2f%%, "
                      "train samples: %.2f (%.2f%%), "
                      "val accuracy: %.2f%%, "
                      "val samples: %.2f (%.2f%%)" % (epoch + 1,
                                                      self.NUM_EPOCHS,
                                                      duration,
                                                      100. * train_accuracy,
                                                      train_steps,
                                                      100. * train_steps / self.SEQUENCE_LENGTH,
                                                      100. * val_accuracy,
                                                      val_steps,
                                                      100. * val_steps / self.SEQUENCE_LENGTH))
                self.logger.info("Absolute losses: entropy: %.3f, budget: %.3f, surprisal: %.3f." % (
                loss_abs[0], loss_abs[1], loss_abs[2]))
                self.logger.info("Percentage losses: entropy: %.2f%%, budget: %.2f%%, surprisal: %.2f%%.\n" % (
                    loss_perc[0], loss_perc[1], loss_perc[2]))
                # print(f"entropy: {loss_plt[epoch, :, 0].mean()}, budget: {loss_plt[epoch, :, 1].mean()}, surprisal: {loss_plt[epoch, :, 2].mean()}.")

                # if EARLY_STOPPING:

            # Training curve for epochs

        except KeyboardInterrupt:
            self.logger.info("Training was interrupted")
            pass

        try:
            plt.plot(train_loss_plt, label='Training loss')
            plt.plot(val_loss_plt, label='Validation loss')
            plt.title(f"Max accuracy {np.max(val_acc_df)}")
            plt.legend()
            plt.savefig(f"{self.FOLDER}/{FILE_NAME}.png")
            plt.clf()

            plt.plot(loss_plt[:, :, 0].flatten(), label='Entropy loss')
            plt.plot(loss_plt[:, :, 1].flatten(), label='Budget loss')
            plt.plot(loss_plt[:, :, 2].flatten(), label='Surprisal loss')
            plt.title("Training losses over time")
            plt.legend()
            plt.savefig(f"{self.FOLDER}/losses_b{self.COST_PER_SAMPLE}_s{self.SURPRISAL_COST}.png")
            plt.clf()
        except:
            self.logger.info("Could not create figures")
            pass

        try:
            df_dict = self.CONFIG_DICT
            df_dict['val_acc'] = val_acc_df
            df_dict['val_updates'] = val_update_df
            df_dict['train_acc'] = train_acc_df
            df_dict['train_updates'] = train_update_df
            df = pd.DataFrame(df_dict)
            csv_loc = '../csvs'
            if not os.path.exists(csv_loc):
                os.makedirs(csv_loc)
            df.to_csv(f"{csv_loc}/{FILE_NAME}.csv")
        except:
            self.logger.info("Could not create csvs")
            pass

        sess.close()
        tf.reset_default_graph()

def get_embedding_dicts(embedding_length):
    PROBS_FILE = 'util/probs.pkl'
    probs_dict = pickle.load(open(PROBS_FILE, 'rb'))

    embedding_dict = {}
    f = open(f'glove.6B.{str(embedding_length)}d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = coefs
    f.close()
    print('Total %s word vectors.' % len(embedding_dict))
    return embedding_dict, probs_dict

def main(argv=None):
    create_generic_flags()
    EMBEDDING_DICT, PROBS_DICT = get_embedding_dicts(embedding_length=50)
    # tf.app.flags.DEFINE_string('data_path', '../data', "Path where the MNIST data will be stored.")
    FLAGS = tf.app.flags.FLAGS
    command_configs = {
        'learning_rate': FLAGS.learning_rate,
        'epochs': FLAGS.epochs,
        'batch_size': FLAGS.batch_size,
        'hidden_units': FLAGS.rnn_cells,
        'cost_per_sample': FLAGS.cost_per_sample,
        'surprisal_cost': FLAGS.surprisal_influence,
        'folder': f'../LR{FLAGS.learning_rate}_BS{FLAGS.batch_size}_HU{FLAGS.rnn_cells}_CPS{FLAGS.cost_per_sample}_SC{FLAGS.surprisal_influence}'
    }
    net = SkipRNN(command_configs, emb_dict = EMBEDDING_DICT, probs_dict=PROBS_DICT)
    print_setup()
    net.train()


if __name__ == '__main__':
    tf.app.run()
