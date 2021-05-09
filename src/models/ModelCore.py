import tensorflow as tf

from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    ModelCheckpoint
)

from src.models.ModelVGG import ModelVGG
from src.models.ModelResnet50 import ModelResnet50
from src.models.OwnModel import OwnModel

class ModelCore():
	def __init__(self, modelType, shape, classes, learning_rate):

		if modelType == 0:
			modelFactory = ModelVGG()
		elif modelType == 1:
			modelFactory = ModelResnet50()
		elif modelType == 2:
			modelFactory = OwnModel()

		print("building model: ", modelFactory.get_name())
			
		model = modelFactory.build_model(shape, classes)

		if learning_rate > 0:
			modelFactory.compile(model, learning_rate)

		print(model.summary())

		self.model = model

	def train(self, train, test, checkpointPath, epochs = 100, steps_per_epoch = 100, validation_steps = 10, save_freq = 20):

		callbacks = [
			ReduceLROnPlateau(verbose=0),
			ModelCheckpoint(checkpointPath + '/model_checkpoint_{epoch}.tf', verbose=1, period=save_freq, save_weights_only=True)
		]

		self.model.fit(train, epochs=epochs, callbacks=callbacks, validation_data=test, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps)

	def save_weights(self, path):
		self.model.save_weights(path)

	def load_model(self, weights):
		self.model.load_weights(weights)
		return self.model