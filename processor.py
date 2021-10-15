import nltk
import keras
import numpy as np
import pickle5 as pickle 
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing import sequence

class Processor:
	def __init__(self):
		self.maxlen = 35
		self.tokenizer = self.load_tokenizer()
		nltk.download('stopwords')

	def load_tokenizer(self):
		with open('tokenizer.pickle', 'rb') as handle:
			pkl = pickle.load(handle)
		return pkl

	def clean_comment(self, comment):
		regexp = "([a-zA-Z]+(?:’[a-z]+)?)"
		regex_tokenizer = RegexpTokenizer(regexp)
		p = """'!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~'"""
		stop_words = stopwords.words('english')
		stop_words.remove('not')

		comment = regex_tokenizer.tokenize(comment)
		comment = [word.lower() for word in comment]
		comment = [''.join(w for w in word if w not in p) for word in comment]
		comment = [word for word in comment if word not in stop_words]
		return comment

	def process(self, text):
		cleaned_comment = self.clean_comment(text)		
		word_seq_train = self.tokenizer.texts_to_sequences(cleaned_comment)		
		wst = []
		for w in word_seq_train:
			if len(wst) == self.maxlen:
				break
			if len(w) != 0:
				wst.append(*w)		
		word_seq_train = sequence.pad_sequences([wst], maxlen=self.maxlen, padding="post")		
		return word_seq_train