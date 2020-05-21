import pickle
import tensorflow_datasets as tfds
import nltk
import numpy as np
from os import path
nltk.download("punkt")
DATA_DIR = '.../data'

class Embedding:
	def __init__(self):

		self.max_sent_len = 3000 # Value emperically found, longest length was 2809
		self.decoder_file = 'decode.pkl'
		self.encoder_file = 'encode.pkl'
		self.probs_file = 'probs.pkl'

		self.unk_word = 'unk'
		self.pad_word = 'pad_word'

		if path.exists(self.encoder_file) and path.exists(self.decoder_file) and path.exists(self.probs_file):
			self.encoder = pickle.load(open(self.encoder_file, 'rb'))
			self.decoder = pickle.load(open(self.decoder_file, 'rb'))
			self.probs = pickle.load(open(self.probs_file, 'rb'))
		else:
			self.encoder, self.decoder, self.probs = self.train_embedding()

	def train_embedding(self):
		encoder = {self.pad_word: 0.0, self.unk_word: 1.0}
		decoder = {0.0: self.pad_word, 1.0: self.unk_word}
		probs = {self.pad_word: 1, self.unk_word: 1}
		train_data, test_data = tfds.load('imdb_reviews/plain_text', split=(tfds.Split.TRAIN, tfds.Split.TEST), with_info=False, as_supervised=True, data_dir=DATA_DIR)
		total_words = 2 # pad and unknown
		idx = 2.0
		for text, label in tfds.as_numpy(train_data):
			# the example is a tuple (text, label)
			for word in nltk.tokenize.word_tokenize(str(text))[1:-1]:
				total_words += 1
				if not word in encoder.keys():
					encoder[word] = idx
					decoder[idx] = word
					probs[word] = 1
					idx += 1
				else:
					probs[word] += 1
		print(f"The vocabulary size is {total_words}")
		probs = {k: v / total_words for k, v in probs.items()}
		probs[self.pad_word] = 1 - np.finfo(float).eps
		probs[self.unk_word] = np.finfo(float).eps

		pickle.dump(encoder, open(self.encoder_file, 'wb'), protocol=0)
		pickle.dump(decoder, open(self.decoder_file, 'wb'), protocol=0)
		pickle.dump(probs, open(self.probs_file, 'wb'), protocol=0)
		return encoder, decoder, probs

	def get_embeddings(self, data):
		inputs = []
		probs = []
		l = []
		for batch in tfds.as_numpy(data):
			inp_batch = []
			probs_batch =[]
			l_batch = []
			for text, label in batch:
				inp = []
				p = []
				tokens = nltk.tokenize.word_tokenize(str(text))[1:-1]
				while len(tokens) < self.max_sent_len:
					tokens.append(self.pad_word)
				for t in tokens:
					if not t in self.encoder.keys():
						t = self.unk_word
					inp.append(self.encoder[t])
					p.append(self.probs[t])
				inp_batch.append(inp)
				probs_batch.append(p)
				l_batch.append(label)
			inputs.append(inp_batch)
			probs.append(probs_batch)
			l.append(l_batch)
		return inputs, probs, l








