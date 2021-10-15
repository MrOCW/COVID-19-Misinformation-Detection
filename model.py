import keras

class Model:
	def __init__(self):
		self.model = None

	def load(self):
		return keras.models.load_model('model.h5')