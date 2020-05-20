import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
import nltk
import numpy as np
import os.path
from os import path

# tf.enable_eager_execution
nltk.download("punkt")


class Embedding:
	# Data is given as ...
	def __init__(self):

		self.max_sent_len = 3000 # Value emperically found, longest length was 2809
		self.decoder_file = 'decode.pkl'
		self.encoder_file = 'decode.pkl'
		self.probs_file = 'probs.pkl'


		self.UNK_WORD = 'unk'
		self.PAD_WORD = 'pad_word'

		if path.exists(self.encode_file) and path.exists(self.decode_file) and path.exists(self.probs_file):
			self.encoder = pickle.load(open(self.encoder_file, 'rb'))
			self.decoder = pickle.load(open(self.decoder_file, 'rb'))
			self.probs = pickle.load(open(self.probs_file, 'rb'))
		else:
			self.encoder, self.decoder, self.probs = train_embedding()

	def train_embedding(self):
		encoder = {self.PAD_WORD: 0, self.UNK_WORD: 1}
		decoder = {0: self.PAD_WORD , 1: self.UNK_WORD}
		probs = {self.PAD_WORD: 1, self.UNK_WORD: 1}
		train_data, test_data, info = tfds.load('imdb_reviews/plain_text', split=(tfds.Split.TRAIN, tfds.Split.TEST), with_info=True, as_supervised=True)
		total_words = 2 # pad and unknown
		idx = 2
		for text, label in tfds.as_numpy(train_data):
			# the example is a tuple (text, label)
			for word in nltk.tokenize.word_tokenize(str(text))[1:-1]:
				total_words += 1
				if not word in encoder.keys():
					encoder[word] = idx
					decoder[idx] = word
					probs[word] = 1
				else:
					probs[word] += 1

		probs = {k: v / total_words for k, v in probs.iteritems()}
		probs[self.PAD_WORD] = 1 - np.finf(float).epsilon
		probs[self.UNK_WORD] = np.finf(float).epsilon

		pickle.dump(encoder, (self.encoder_file, 'wb'), protocol= 0)
		pickle.dump(decoder, (self.decoder_file, 'wb'), protocol= 0)
		pickle.dump(probs, (self.probs_file, 'wb'), protocol= 0)
		return encoder, decoder, probs

	def get_embeddings(self, data):
		inp = []
		label = []
		for text, label in tfds.as_numpy(data):
			tokens = nltk.tokenize.word_tokenize(str(text))[2:-1]
			while len(tokens) < self.max_sent_len:
				tokens.append(self.pad_word)
			for t in tokens:
				try:
					inp.append(self.encoder[t])
				except Exception:
					print(f"Unknown word {t} was found")
					inp.append(self.encoder[self.UNK_WORD])








