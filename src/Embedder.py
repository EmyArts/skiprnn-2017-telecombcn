import pickle
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow_datasets as tfds


class Embedding:
	# Data is given as ...
	def __init__(self, pretrained):

		self.decoder_file = 'decode.pkl'
		self.encoder_file = 'decode.pkl'

		self.UNK_WORD = 'unk'
		self.PAD_WORD = '<pad>'

		if pretrained:
			self.encoder = pickle.load(open(self.encoder_file, 'rb'))
			self.decoder = pickle.load(open(self.decoder_file, 'rb'))
		else:
			self.encoder, self.decoder = train_embedding(data)

	def train_embedding(self, embedding_data):
		encoder = {}
		decoder = {}
		train_data, test_data, info = tfds.load('imdb_reviews/plain_text', split=(tfds.Split.TRAIN, tfds.Split.TEST), with_info=True, as_supervised=True)
		for word, _ in train_data:






