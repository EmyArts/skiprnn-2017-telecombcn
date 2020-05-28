from gensim.test.utils import common_texts
from gensim.utils import tokenize
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from os import path, getcwd
import numpy as np
import pickle
import nltk
import tensorflow
import tensorflow_datasets as tfds
import multiprocessing
nltk.download("punkt")
DATA_DIR = '.../data'

class Gensim_Embedding:

	def __init__(self):
		# self.model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
		self.max_sent_len = 2520  # Value emperically found, longest length was 2514
		self.vec_len = 50
		self.glove_input_file = 'glove.6B.50d.txt'
		self.encoder_file = 'glove.word2vec'
		# self.decoder_file = 'decode.pkl'
		# self.encoder_file = 'encode.pkl'
		self.probs_file = 'probs.pkl'
		self.vocab_size = 3000 * 25000

		self.unk_word = 'unk'
		self.pad_word = 'pad_word'

		if path.exists(self.encoder_file) and path.exists(self.probs_file):
			# if path.exists(self.encoder_file) and path.exists(self.decoder_file) and path.exists(self.probs_file):
			print("\nUsing pkl files for embedding\n")
			# self.encoder = pickle.load(open(self.encoder_file, 'rb'))
			# self.decoder = pickle.load(open(self.decoder_file, 'rb'))
			self.probs = pickle.load(open(self.probs_file, 'rb'))
		else:
			# self.encoder, self.decoder, self.probs = self.train_embedding()
			glove2word2vec(self.glove_input_file, self.encoder_file)
			self.probs = self.train_embedding()
		self.encoder = KeyedVectors.load_word2vec_format(self.encoder_file, binary=False)

	def train_embedding(self):
		print("\nTraining embedding\n")
		#decoder = {0.0: self.pad_word, 1.0: self.unk_word}
		probs = {self.pad_word: 1, self.unk_word: 1}
		data = tfds.load('imdb_reviews/plain_text', split='unsupervised', data_dir=DATA_DIR)
		total_words = 2 # pad and unknown
		idx = 2.0
		max_len = 0
		for text in tfds.as_numpy(data):
			# the example is a tuple (text, label)
			tokens = list(tokenize(str(text), lowercase=True))[3:]
			# if len(tokens) > max_len:
			# 	 print(len(tokens))
			for idx, word in enumerate(tokens):
				total_words += 1
				if not word in probs.keys():
					#encoder[word] = idx
					#decoder[idx] = word
					probs[word] = 1
					idx += 1
				else:
					probs[word] += 1
			if idx > max_len:
				max_len = idx
		print(f"The vocabulary size is {total_words}")
		print(f"The maximum length of a review is {max_len}")
		self.vocab_size = total_words
		probs = {k: v / total_words for k, v in probs.items()}
		probs[self.pad_word] = 1 - np.finfo(np.float32).eps
		probs[self.unk_word] = np.finfo(np.float32).eps

		#pickle.dump(encoder, open(self.encoder_file, 'wb'), protocol=0)
		#pickle.dump(decoder, open(self.decoder_file, 'wb'), protocol=0)
		pickle.dump(probs, open(self.probs_file, 'wb'), protocol=0)
		#return encoder, decoder, probs
		return probs

	def get_embeddings(self, data):
		print("Creating embeddings.")
		inputs = []
		ps = []
		l = []
		for text, label in tfds.as_numpy(data):
			inp = np.zeros((self.max_sent_len, self.vec_len))
			p = np.full((self.max_sent_len, self.vec_len), self.probs[self.pad_word])
			tokens = list(tokenize(str(text), lowercase=True))[3:]
			for i, t in enumerate(tokens):
				try:
					inp[i] = self.encoder[t]
					p[i] = np.full(self.vec_len, self.probs[t])
				except Exception:
					pass
			inputs.append(inp)
			ps.append(p)
			l.append(label)
		return tf.data.Dataset.from_tensor_slices((np.array(inputs, dtype=np.float32), np.array(ps, dtype=np.float32), np.array(l)))