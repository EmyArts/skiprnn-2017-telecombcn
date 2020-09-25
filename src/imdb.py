"""
Train RNN models on sequential MNIST, where inputs are processed pixel by pixel.

Results should be reported by evaluating on the test set the model with the best performance on the validation set.
To avoid storing checkpoints and having a separate evaluation script, this script evaluates on both validation and
test set after every epoch.
"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import time
import datetime
import pickle
import logging
from threading import Thread


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
from src.util.graph_definition import create_model, split_rnn_outputs, compute_budget_loss, compute_surprisal_loss, \
    compute_gradients, create_generic_flags
from src.util.misc import compute_used_samples, print_setup


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
        self.FILE_NAME = config_dict['file_name']
        self.EARLY_STOPPING = config_dict['early_stopping']
        self.TRIAL = config_dict['trial']

        # Constants
        self.OUTPUT_SIZE = 2
        self.SEQUENCE_LENGTH = 625  # Found emperically 95th percentile
        self.EMBEDDING_LENGTH = 50

        # Originalli 25k for training and 25k for testing -> 15k for validation and 10k for testing
        # Keras used 15k for training, 10k for validation out of the training set and 25k for testing later
        # self.TRAIN_SAMPLES = 9984  # Colab
        # self.VAL_SAMPLES = 4992
        # self.TEST_SAMPLES = 9984
        # self.TRAIN_SAMPLES = 14976  # Server
        # self.VAL_SAMPLES = 9984
        # self.TEST_SAMPLES = 14976
        self.TRAIN_SAMPLES = 320  # Debug
        self.VAL_SAMPLES = 192
        self.TEST_SAMPLES = 320
        # TRAIN and VAL samples should always sum up to 25k

        # TRAIN_SAMPLES = info.splits[tfds.Split.TRAIN].num_examples
        # TEST_SAMPLES = info.splits[tfds.Split.TEST].num_examples

        self.ITERATIONS_PER_EPOCH = int(self.TRAIN_SAMPLES / self.BATCH_SIZE)
        self.VAL_ITERS = int(self.VAL_SAMPLES / self.BATCH_SIZE)
        self.TEST_ITERS = int(self.TEST_SAMPLES / self.BATCH_SIZE)
        # ITERATIONS_PER_EPOCH = int(5000/BATCH_SIZE)
        # TEST_ITERS = int(5000/BATCH_SIZE)
        self.TEST_EMBEDDING_MATRIX = np.zeros(
            (self.TEST_ITERS, self.BATCH_SIZE, self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH), dtype=np.float32)
        # Load data
        self.imdb_builder = tfds.builder('imdb_reviews/plain_text', data_dir='../data')
        self.imdb_builder.download_and_prepare()
        # info = imdb_builder.info

        # Setting up logger
        # os.mkdir(self.FOLDER)
        # logging.basicConfig(filename=f"{self.FOLDER}/log.log", filemode='w', format='%(asctime)s - %(message)s',
        #             level=logging.INFO)
        if not os.path.exists('../net_logs'):
            os.makedirs('../net_logs')
        self.logger = logging.getLogger("Net Logger")
        fh = logging.FileHandler(f"../net_logs/{self.FILE_NAME}.log")
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(fh)

        self.logger.info(
            f"\n\n{self.FILE_NAME}\n"
            f"\nLearning rate: {self.LEARNING_RATE}\nBatch size: {self.BATCH_SIZE}\nHidden units: {self.HIDDEN_UNITS}"
            f"\nCost per sample: {self.COST_PER_SAMPLE}\nSurprisal cost {self.SURPRISAL_COST}\n\n")
        # self.logger.basicConfig(filename=f"{self.FOLDER}/log.log", filemode='w', format='%(asctime)s - %(message)s', level=logging.INFO)


    def input_fn(self, split):
        # Reset datasets once done debugging
        val_split = f'train[{self.TRAIN_SAMPLES}:{self.TRAIN_SAMPLES + self.VAL_SAMPLES}]'
        # test_split = f'test[{TRAIN_SAMPLES:{TEST_SAMPLES}]'
        # valid_split = f'test[{TEST_SAMPLES}:]'
        if split == 'train':
            tot_len = math.ceil(self.TRAIN_SAMPLES / self.BATCH_SIZE)
            data = self.imdb_builder.as_dataset(as_supervised=True, split=f'train[:{self.TRAIN_SAMPLES}]')
            # This will be the ITERATIONS_PER_EPOCH
            # print("Total amount of training samples: " + str(len(list(dataset))))
            # print("Total amount of validation samples: " + str(len(list(dataset))))
        elif split == 'val':
            data = self.imdb_builder.as_dataset(as_supervised=True, split=val_split)
            tot_len = math.ceil(self.VAL_SAMPLES / self.BATCH_SIZE)  # This will be VAL_ITERS
            # print("Total amount of test samples: " + str(len(list(dataset))))
        elif split == 'test':
            data = self.imdb_builder.as_dataset(as_supervised=True, split=f'test[:{self.TEST_SAMPLES}]')
            tot_len = math.ceil(self.TEST_SAMPLES / self.BATCH_SIZE)
        else:
            raise ValueError()

        # print(f"Vector for unknonw words is {embeddings_index.get('unk')}")
        embedding_matrix = np.zeros((tot_len, self.BATCH_SIZE, self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH), dtype=np.float32)
        probs_matrix = np.ones((tot_len, self.BATCH_SIZE, self.SEQUENCE_LENGTH, 1), dtype=np.float32)
        mask = np.zeros((tot_len, self.BATCH_SIZE, self.SEQUENCE_LENGTH, 1), dtype=np.float32)
        labels = np.zeros((tot_len, self.BATCH_SIZE), dtype=np.int64)
        line_index = 0
        batch_index = 0
        c_unk = 0
        word_count = 0
        entry = 0
        for text, label in tfds.as_numpy(data):
            labels[batch_index][entry] = label
            # print(label)
            tokens = list(tokenize(str(text), lowercase=True))[3:]
            if len(tokens) >= self.SEQUENCE_LENGTH:
                tokens = tokens[:self.SEQUENCE_LENGTH]
            for i, t in enumerate(tokens):
                mask[batch_index][entry][i] = 1
                embedding_vector = self.EMBEDDING_DICT.get(t)
                prob = self.PROBS_DICT.get(t)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[batch_index][entry][i] = embedding_vector
                    probs_matrix[batch_index][entry][i] = prob
                else:
                    c_unk += 1
                    embedding_matrix[batch_index][entry][i] = self.EMBEDDING_DICT.get("unk_word")
                    probs_matrix[batch_index][entry][i] = self.PROBS_DICT.get("unk_word")
                word_count += 1
            line_index += 1
            entry = line_index % self.BATCH_SIZE
            if entry == 0:
                batch_index += 1
        if split == 'test':
            self.TEST_EMBEDDING_MATRIX = embedding_matrix
        self.logger.info(f"{c_unk} words out of {word_count} total words unknown for {split} set.")
        print(f"{c_unk} words out of {word_count} total words unknown for {split} set.")

        # inputs = {'text': text, 'labels': labels, 'iterator_init_op': iterator_init_op}
        print(
            f"\n\n Input shape is {embedding_matrix.shape},  labels shape is {labels.shape}, probs shape is {probs_matrix.shape}")
        # np.expand_dims(probs_matrix, axis=-1)
        return embedding_matrix, labels, probs_matrix, mask

    # print_samples = tf.Print(samples, [samples], "\nSamples are: \n")

    # def train(hyper_params = BASE_CONF):
    def train(self):
        samples = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH],
                                 name='Samples')  # (batch, time, in)
        ground_truth = tf.placeholder(tf.int64, shape=[self.BATCH_SIZE], name='GroundTruth')
        probs = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH, 1], name='Probs')
        mask = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE, self.SEQUENCE_LENGTH, 1], name='padding_mask')

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
        printer_lab = tf.cond(
            tf.math.reduce_any(tf.logical_or(tf.equal(tf.zeros_like(ground_truth), ground_truth),
                                             tf.equal(tf.ones_like(ground_truth), ground_truth))),
            lambda: tf.no_op(),
            lambda: tf.print("Found a label out of range: ", [ground_truth]))
        with tf.control_dependencies([printer_lab]):
            cross_entropy_per_sample = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                                      labels=ground_truth)
        # cross_entropy_per_sample = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=ground_truth)
        # max_ce = tf.math.maximum(cross_entropy_per_sample)
        # median_ce = tf.math.median(cross_entropy_per_sample)
        # printer_max = tf.Print(max_ce, [max_ce], "The maximum cross entropy is ")
        # printer_median = tf.Print(median_ce, [median_ce], "The median cross entropy is ")
        printer_Nan = tf.cond(tf.math.reduce_any(tf.math.is_nan(cross_entropy_per_sample)),
                              lambda: tf.print("Found NaN in entropy loss", output_stream=sys.stderr),
                              lambda: tf.no_op())
        with tf.control_dependencies([printer_Nan]):
            cross_entropy = tf.reduce_mean(
                tf.boolean_mask(cross_entropy_per_sample, tf.is_finite(cross_entropy_per_sample)))

            # tf.where(tf.math.is_nan(cross_entropy_per_sample),
            #          tf.ones(cross_entropy_per_sample.get_shape()),
            #          cross_entropy_per_sample))

        # Compute accuracy
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, ground_truth), tf.float32))

        # updated_states = tf.boolean_mask(updated_states, mask)

        # Compute loss for each updated state
        budget_loss = compute_budget_loss('skip_lstm', cross_entropy, updated_states, self.COST_PER_SAMPLE, mask)
        # printer_Nan = tf.cond(tf.math.reduce_any(tf.math.is_nan(budget_loss)),
        #                       lambda: tf.print("Found NaN in budget loss"), lambda: tf.no_op())
        # with tf.control_dependencies([printer_Nan]):
        #     budget_loss = tf.where(tf.math.is_nan(budget_loss),
        #                            tf.ones(budget_loss.get_shape()),
        #                            budget_loss)

        # Compute loss for the amount of surprisal
        surprisal_loss = compute_surprisal_loss('skip_lstm', cross_entropy, updated_states, probs, self.SURPRISAL_COST,
                                                mask)
        # Avoid encouraging to not skip.
        # printer_Nan = tf.cond(tf.math.reduce_any(tf.math.is_nan(surprisal_loss)),
        #                       lambda: tf.print("Found NaN in surprisal loss"), lambda: tf.no_op())
        # with tf.control_dependencies([printer_Nan]):
        #     surprisal_loss = tf.where(tf.math.logical_or(tf.equal(surprisal_loss, tf.zeros_like(surprisal_loss)),
        #                                                  tf.math.is_nan(surprisal_loss)), tf.ones_like(surprisal_loss),
        #                               surprisal_loss)

        loss = cross_entropy + budget_loss + surprisal_loss
        loss = tf.reshape(loss, [])

        loss = tf.where(tf.is_nan(loss), tf.ones_like(loss), loss)

        # Optimizer
        opt, grads_and_vars = compute_gradients(loss, self.LEARNING_RATE, 1)  # used to be 1 is for gradient clipping
        train_fn = opt.apply_gradients(grads_and_vars)

        sess = tf.Session()

        # log_dir = os.path.join(self.LOG_DIR, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        # val_writer = tf.summary.FileWriter(log_dir + '/validation')

        # Initialize weights
        sess.run(tf.global_variables_initializer())

        # Results
        train_loss_plt = np.zeros((self.NUM_EPOCHS))
        loss_plt = np.zeros((self.NUM_EPOCHS, self.ITERATIONS_PER_EPOCH, 3))
        val_acc_df = np.zeros((self.NUM_EPOCHS))
        train_acc_df = np.zeros((self.NUM_EPOCHS))
        # test_acc_df = np.zeros((self.NUM_EPOCHS))
        train_update_df = np.zeros((self.NUM_EPOCHS))
        val_update_df = np.zeros((self.NUM_EPOCHS))
        # test_update_df = np.zeros((self.NUM_EPOCHS))
        # test_time_df = np.zeros((self.NUM_EPOCHS))

        read_embs = np.zeros((self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH))
        non_read_embs = np.zeros((self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH))
        read_surps = np.ones((self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH))
        non_read_surps = np.ones((self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH))

        # FILE_NAME = f'hu{self.HIDDEN_UNITS}_bs{self.BATCH_SIZE}_lr{self.LEARNING_RATE}_b{self.COST_PER_SAMPLE}_s{self.SURPRISAL_COST}_t{self.TRIAL}'

        try:
            train_matrix, train_labels, train_probs, train_mask = self.input_fn(split='train')
            val_matrix, val_labels, val_probs, val_mask = self.input_fn(split='val')
            test_matrix, test_labels, test_probs, test_mask = self.input_fn(split='test')

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
                                              probs: train_probs[iteration],
                                              mask: train_mask[iteration]
                                              })
                    train_accuracy += out[2]
                    train_loss += out[1]
                    loss_plt[epoch][iteration] = out[4:]  # entropy, budget, surprisal
                    if out[3] is not None:
                        train_steps += compute_used_samples(out[3] * train_mask[iteration])
                    else:
                        train_steps += np.count_nonzero(train_mask[iteration])

                duration = time.time() - start_time

                train_accuracy /= self.ITERATIONS_PER_EPOCH
                train_loss /= self.ITERATIONS_PER_EPOCH
                train_steps /= (np.count_nonzero(train_mask) / self.BATCH_SIZE)
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
                                                                                     probs: val_probs[iteration],
                                                                                     mask: val_mask[iteration]
                                                                                 })
                    val_accuracy += val_iter_accuracy
                    val_loss += val_iter_loss
                    if val_used_inputs is not None:
                        val_steps += compute_used_samples(val_used_inputs * val_mask[iteration])
                    else:
                        val_steps += np.count_nonzero(val_mask[iteration])
                val_accuracy /= self.VAL_ITERS
                val_loss /= self.VAL_ITERS
                val_steps /= (np.count_nonzero(val_mask) / self.BATCH_SIZE)
                val_acc_df[epoch] = val_accuracy
                val_update_df[epoch] = val_steps

                # val_writer.add_summary(scalar_summary('accuracy', val_accuracy), epoch)
                # val_writer.add_summary(scalar_summary('loss', val_loss), epoch)
                # val_writer.add_summary(scalar_summary('used_samples', val_steps / self.SEQUENCE_LENGTH), epoch)
                # val_writer.flush()

                # print("Epoch %d/%d, "
                #       "duration: %.2f seconds, "
                #       "train accuracy: %.2f%%, "
                #       "train samples: %.2f (%.2f%%), "
                #       "val accuracy: %.2f%%, "
                #       "val samples: %.2f (%.2f%%)" % (epoch + 1,
                #                                        self.NUM_EPOCHS,
                #                                        duration,
                #                                        100. * train_accuracy,
                #                                        train_steps,
                #                                        100. * train_steps / self.SEQUENCE_LENGTH,
                #                                        100. * val_accuracy,
                #                                        val_steps,
                #                                        100. * val_steps / self.SEQUENCE_LENGTH))
                #

                # print("Absolute losses: entropy: %.3f, budget: %.3f, surprisal: %.3f." % (loss_abs[0], loss_abs[1], loss_abs[2]))
                #
                # print("Percentage losses: entropy: %.2f%%, budget: %.2f%%, surprisal: %.2f%%.\n" % (loss_perc[0], loss_perc[1], loss_perc[2]))

                loss_abs = loss_plt[epoch].mean(axis=0)
                loss_perc = np.divide(loss_abs, (loss_abs.sum())) * 100

                self.logger.info("Epoch %d/%d, "
                                 "duration: %.2f seconds, "
                                 "train accuracy: %.2f%%, "
                                 "train samples: %.2f%%, "
                                 "val accuracy: %.2f%%, "
                                 "val samples: %.2f%%" % (epoch + 1,
                                                          self.NUM_EPOCHS,
                                                          duration,
                                                          100. * train_accuracy,
                                                          100. * train_steps,
                                                          100. * val_accuracy,
                                                          100. * val_steps))
                self.logger.info("Absolute losses: entropy: %.3f, budget: %.3f, surprisal: %.3f." % (
                    loss_abs[0], loss_abs[1], loss_abs[2]))
                self.logger.info("Percentage losses: entropy: %.2f%%, budget: %.2f%%, surprisal: %.2f%%." % (
                    loss_perc[0], loss_perc[1], loss_perc[2]))
                # print(f"entropy: {loss_plt[epoch, :, 0].mean()}, budget: {loss_plt[epoch, :, 1].mean()}, surprisal: {loss_plt[epoch, :, 2].mean()}.")
                analysis_update = val_accuracy + 1e-4 > val_acc_df.max()
                if analysis_update:
                    self.logger.info("Updating Analysis")
                    read_embs = np.zeros(
                        (self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH))
                    non_read_embs = np.zeros(
                        (self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH, self.EMBEDDING_LENGTH))
                    read_surps = np.full((self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH), -1)
                    non_read_surps = np.full((self.TEST_ITERS * self.BATCH_SIZE * self.SEQUENCE_LENGTH), -1)

                test_accuracy, test_loss, test_steps, t = 0, 0, 0, 0
                for iteration in range(self.TEST_ITERS):
                    t0 = time.time()
                    test_iter_accuracy, test_iter_loss, test_used_inputs = sess.run([accuracy, loss, updated_states],
                                                                                    feed_dict={
                                                                                        samples: test_matrix[iteration],
                                                                                        ground_truth: test_labels[
                                                                                            iteration],
                                                                                        probs: test_probs[iteration],
                                                                                        mask: test_mask[iteration]
                                                                                    })
                #     t += time.time() - t0
                #     test_accuracy += test_iter_accuracy
                #     test_loss += test_iter_loss
                #     if test_used_inputs is not None:
                #         test_steps += compute_used_samples(test_used_inputs * test_mask[iteration])
                #         if analysis_update:
                #             try:
                #                 re, nre, rs, nrs = stats_used_samples(test_used_inputs, test_matrix[iteration],
                #                                                       test_probs[iteration], test_mask[iteration])
                #                 read_embs[
                #                 self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH: self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH + len(
                #                     re)] = re
                #                 non_read_embs[
                #                 self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH: self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH + len(
                #                     nre)] = nre
                #                 read_surps[
                #                 self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH: self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH + len(
                #                     rs.flatten())] = rs.flatten()  # take out flatten but should not be the problem
                #                 non_read_surps[
                #                 self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH: self.BATCH_SIZE * iteration * self.SEQUENCE_LENGTH + len(
                #                     nrs.flatten())] = nrs.flatten()
                #             except Exception as e:
                #                 self.logger.info("Could not update analysis")
                #                 self.logger.error(e)
                #                 pass
                #     else:
                #         test_steps += np.count_nonzero(test_mask[iteration])
                #
                # test_accuracy /= self.TEST_ITERS
                # test_loss /= self.TEST_ITERS
                # test_steps /= (np.count_nonzero(test_mask) / self.BATCH_SIZE)
                # test_time_df[epoch] = t
                # test_acc_df[epoch] = test_accuracy
                # test_update_df[epoch] = test_steps
                #
                # self.logger.info("Test time: %.2f seconds, "
                #                  "test accuracy: %.2f%%, "
                #                  "test samples: %.2f%%.\n"
                #                  % (test_time_df[epoch],
                #                     100. * test_accuracy,
                #                     100. * test_steps))

                if self.EARLY_STOPPING and epoch > 15:
                    if epoch == 16:
                        best_accuracy = val_acc_df.max()
                        best_idx = val_acc_df.argmax()
                    if best_accuracy < val_acc_df[epoch] + 1e-4:
                        best_accuracy = val_acc_df[epoch]
                        best_idx = epoch
                    elif best_idx + 15 < epoch:
                        val_update_df = val_update_df[:epoch]
                        val_acc_df = val_acc_df[:epoch]
                        train_acc_df = train_acc_df[:epoch]
                        train_update_df = train_update_df[:epoch]
                        loss_plt = loss_plt[:epoch]
                        # test_acc_df = test_acc_df[:epoch]
                        # test_update_df = test_update_df[:epoch]
                        # test_time_df = test_time_df[:epoch]
                        self.logger.info("Training was interrupted with early stopping")
                        break



        except KeyboardInterrupt:
            self.logger.info("Training was interrupted manually")
            pass

        try:
            df_dict = self.CONFIG_DICT
            df_dict['val_acc'] = val_acc_df
            df_dict['val_updates'] = val_update_df
            df_dict['train_acc'] = train_acc_df
            df_dict['train_updates'] = train_update_df
            # df_dict['test_acc'] = test_acc_df
            # df_dict['test_updates'] = test_update_df
            # df_dict['test_time'] = test_time_df
            loss_plt_mean = loss_plt.mean(axis=1).transpose()
            df_dict['entropy_loss'] = loss_plt_mean[0]
            df_dict['budget_loss'] = loss_plt_mean[1]
            df_dict['surprisal_loss'] = loss_plt_mean[2]
            df = pd.DataFrame(df_dict)
            df.drop(columns=['epochs', 'file_name'], inplace=True)
            csv_loc = '../csvs'
            if not os.path.exists(csv_loc):
                os.makedirs(csv_loc)
            df.to_csv(f"{csv_loc}/{self.FILE_NAME}.csv")
        except Exception as e:
            print(e)
            self.logger.info("Could not create csvs")
            pass

        ## Saving analysis statistics
        try:
            analysis_loc = '../analysis'
            if not os.path.exists(analysis_loc):
                os.makedirs(analysis_loc)
            print("Read words")
            read_words = get_words_from_embedding(self.EMBEDDING_DICT, self.TEST_EMBEDDING_MATRIX, read_embs)
            print("Skipped words")
            non_read_words = get_words_from_embedding(self.EMBEDDING_DICT, self.TEST_EMBEDDING_MATRIX, non_read_embs)
            pickle.dump(read_words, open(f"{analysis_loc}/{self.FILE_NAME}_read_vocab.pkl", 'wb'), protocol=0)
            pickle.dump(non_read_words, open(f"{analysis_loc}/{self.FILE_NAME}_non_read_vocab.pkl", 'wb'), protocol=0)
            read_surps = np.vstack(read_surps).flatten()
            non_read_surps = np.vstack(non_read_surps).flatten()
            np.save(open(f"{analysis_loc}/{self.FILE_NAME}_read_surprisals.npy", 'wb'), read_surps[read_surps >= 0])
            np.save(open(f"{analysis_loc}/{self.FILE_NAME}_non_read_surprisals.npy", 'wb'),
                    non_read_surps[non_read_surps >= 0])

        except Exception as e:
            print(e)
            self.logger.info("Something went wrong when reporting analysis results")
            pass

        sess.close()
        tf.reset_default_graph()

def get_embedding_dicts(embedding_length):
    PROBS_FILE = 'util/probs.pkl'
    probs_dict = pickle.load(open(PROBS_FILE, 'rb'))
    probs_dict["unk_word"] = 1

    embedding_dict = {}
    f = open(f'glove.6B.{str(embedding_length)}d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_dict[word] = coefs
    f.close()
    embedding_dict["unk_word"] = [-0.12920076, -0.28866628, -0.01224866, -0.05676644, -0.20210965, -0.08389011,
                                  0.33359843, 0.16045167, 0.03867431, 0.17833012, 0.04696583, -0.00285802,
                                  0.29099807, 0.04613704, -0.20923874, -0.06613114, -0.06822549, 0.07665912,
                                  0.3134014, 0.17848536, -0.1225775, -0.09916984, -0.07495987, 0.06413227,
                                  0.14441176, 0.60894334, 0.17463093, 0.05335403, -0.01273871, 0.03474107,
                                  -0.8123879, -0.04688699, 0.20193407, 0.2031118, -0.03935686, 0.06967544,
                                  -0.01553638, -0.03405238, -0.06528071, 0.12250231, 0.13991883, -0.17446303,
                                  -0.08011883, 0.0849521, -0.01041659, -0.13705009, 0.20127155, 0.10069408,
                                  0.00653003, 0.01685157]
    print('Total %s word vectors.' % len(embedding_dict))
    return embedding_dict, probs_dict


def get_words_from_embedding(embedding_dict, embeddings):
    # print("Hey from get words form embeddings!")
    vocab = {}
    # print(f"embedding dictionary values: {list(embedding_dict.values())[:5]}\n")
    inv_embedding_dict = {tuple(v): k for k, v in embedding_dict.items()}
    keys = inv_embedding_dict.keys()
    # print(f"inverse embedding dictionary keys: {list(keys[:5]}\n")
    for emb in embeddings:
        # print(f"Embedding: {emb}\n")
        # pos = np.where(np.all(embedding_matrix == emb, axis=1))
        # print(pos)
        # assert len(pos) == 1
        if tuple(emb) in keys:
            word = inv_embedding_dict[tuple(emb)]
            # print(f"Word found {word}")
            if word in vocab.keys():
                vocab[word] += 1
            else:
                vocab[word] = 1
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1], reverse=True)}
    print({k: vocab[k] for k in list(vocab.keys())[:10]})
    return vocab


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
        'file_name': f'LR{FLAGS.learning_rate}_BS{FLAGS.batch_size}_HU{FLAGS.rnn_cells}_CPS{FLAGS.cost_per_sample}_SC{FLAGS.surprisal_influence}',
        'early_stopping': (FLAGS.early_stopping == 'yes'),
        'trial': 0
    }
    net = SkipRNN(command_configs, emb_dict = EMBEDDING_DICT, probs_dict=PROBS_DICT)
    print_setup()
    net.train()


if __name__ == '__main__':
    tf.app.run()
