import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Embedding
from keras.layers import Flatten
class Model:
	def __init__(self):
		self.model = None

	def load(self,weights: str):
		return keras.models.load_model(weights)
