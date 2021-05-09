import tensorflow as tf
from src.models.base.Loss import Loss
from src.metrics.Metrics import Metrics

class BaseModel():
	def get_name(self):
		return "not implemented"

	def build_model(self, shape, numClasses, weights = ''):
		pass

	def compile(self, model, learning_rate = 0.001):
		optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
		model.compile(loss=self.Loss(), optimizer=optimizer, metrics=self.Accuracy())

	def Loss(self):
		return [Loss.LossIOU]

	def Accuracy(self):
		return [Metrics.AccuracyIoU, Metrics.AccuracyClassification]