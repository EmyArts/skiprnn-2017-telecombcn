import pickle
import tensorflow_datasets as tfds
import tensorflow as tf
import nltk
from gensim.models import Word2Vec
from gensim.utils import tokenize
import numpy as np
from os import path
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import multiprocessing

nltk.download("punkt")
DATA_DIR = '.../data'


class Embedding:
	def __init__(self):

		self.max_sent_len = 2520  # Value emperically found, longest length was 2514
		self.vec_len = 50
		# self.decoder_file = 'decode.pkl'

		# Files for avoiding initialization between calls.
		self.encoder_file = 'encode.pkl'
		self.probs_file = 'probs.pkl'
		self.matrix_file = 'emb_matrix.npy'

		# Files for pre trained embedding.
		self.glove_input_file = 'glove.6B.50d.txt'
		self.model_file = 'glove.word2vec'

		self.unk_word = 'unk'
		self.pad_word = 'pad_word'

		if path.exists(self.encoder_file) and path.exists(self.probs_file) and path.exists(self.matrix_file) :
			# if path.exists(self.encoder_file) and path.exists(self.decoder_file) and path.exists(self.probs_file):
			print("\nUsing pkl files for embedding\n")
			self.encoder = pickle.load(open(self.encoder_file, 'rb'))
			# self.decoder = pickle.load(open(self.decoder_file, 'rb'))
			self.probs = pickle.load(open(self.probs_file, 'rb'))
			self.emb_matrix = np.load(open(self.matrix_file, 'rb'))
		else:
			# self.encoder, self.decoder, self.probs = self.train_embedding()
			self.encoder, self.probs, self.emb_matrix = self.train_embedding()


	def train_embedding(self):
		print("\nTraining embedding\n")
		encoder = {self.pad_word: 0, self.unk_word: 1}
		# decoder = {0.0: self.pad_word, 1.0: self.unk_word}
		probs = {self.pad_word: 1, self.unk_word: 1}
		data = tfds.load('imdb_reviews/plain_text', split='unsupervised', data_dir=DATA_DIR)
		total_words = 2  # pad and unknown
		entry_count = 1
		max_len = 0
		for text in tfds.as_numpy(data):
			tokens = list(tokenize(str(text), lowercase=True))[3:]
			for idx, word in enumerate(tokens):
				total_words += 1
				if not word in encoder.keys():
					entry_count += 1
					encoder[word] = entry_count
					probs[word] = 1
				else:
					probs[word] += 1
			if idx > max_len:
				max_len = idx
		print(f"The vocabulary size is {entry_count}")
		print(f"The maximum length of a review is {max_len}")
		probs = {k: v / total_words for k, v in probs.items()}
		probs[self.pad_word] = 1 - np.finfo(np.float32).eps
		probs[self.unk_word] = np.finfo(np.float32).eps

		glove2word2vec(self.glove_input_file, self.model_file)
		model = KeyedVectors.load_word2vec_format(self.model_file, binary=False)


		print("Creating matrix")
		skipped_words=0
		emb_matrix = np.zeros((entry_count, self.vec_len), dtype=np.float32)
		for i, word in enumerate(encoder.keys()):
			try:
				emb_matrix[i] = model[word]
			except:
				skipped_words += 1
				pass

		print(f"Skipped {skipped_words} out of {entry_count}")
		np.save(open(self.matrix_file, 'wb'), emb_matrix)
		pickle.dump(encoder, open(self.encoder_file, 'wb'), protocol=0)
		pickle.dump(probs, open(self.probs_file, 'wb'), protocol=0)

		return encoder, probs, emb_matrix


	def get_embeddings(self, data):
		print("Creating embeddings.")
		inputs = []
		ps = []
		l = []
		for text, label in tfds.as_numpy(data):
			inp = np.full(self.max_sent_len, self.encoder[self.pad_word])
			p = np.full(self.max_sent_len, self.probs[self.pad_word])
			tokens = list(tokenize(str(text), lowercase=True))[3:]
			for i, t in enumerate(tokens):
				if not t in self.encoder:
					t = self.unk_word
				inp[i] = self.encoder[t]
				p[i] = self.probs[t]
			inputs.append(inp)
			ps.append(p)
			l.append(label)
		return tf.data.Dataset.from_tensor_slices((np.array(inputs, dtype=np.int64), np.array(ps, dtype=np.float32), np.array(l)))

	def embedding_matrix(self):
		return tf.convert_to_tensor(self.emb_matrix)

	def vector_length(self):
		return self.vec_len