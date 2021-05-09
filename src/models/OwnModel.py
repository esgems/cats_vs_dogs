import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.models.base.BaseModel import BaseModel

class OwnModel(BaseModel):
	def get_name(self):
		return "own model"

	def ConvLayer(self, inp, filters, kernelSize=3, stride=(1, 1)):
		out = layers.Conv2D(filters=filters, kernel_size=(kernelSize, kernelSize), strides=stride, 
			padding='same',
			kernel_regularizer=keras.regularizers.l2(1e-4),
			activation=None)(inp)
		out = layers.LeakyReLU(alpha=0.01)(out)
		# out = layers.BatchNormalization()(out)
		return out

	def ConvLayer1Dim(self, inp, filters, kernelSize=3, stride=(1, 1)):
		out = layers.Conv2D(filters=filters, kernel_size=(kernelSize, 1), strides=stride, 
			padding='same',
			kernel_regularizer=keras.regularizers.l2(1e-4),
			activation=None)(inp)
		out = layers.LeakyReLU(alpha=0.01)(out)
		out = layers.Conv2D(filters=filters, kernel_size=(1, kernelSize), strides=stride, 
			padding='same',
			kernel_regularizer=keras.regularizers.l2(1e-4),
			activation=None)(out)
		out = layers.LeakyReLU(alpha=0.01)(out)
		# out = layers.BatchNormalization()(out)
		return out

	def ResidualLayer(self, inp, filters, kernelSize, newDepth=-1):

		conv1 = self.ConvLayer(inp, filters, kernelSize)
		conv2 = self.ConvLayer(conv1, filters, kernelSize)
		out = layers.Add()([inp, conv2])

		if newDepth > -1:
			pool = layers.MaxPooling2D((2, 2))(out)
			return self.ConvLayer(pool, newDepth, 1)

		return out

	def ResidualLayer2(self, inp, filters, kernelSize, newDepth=-1):
		conv1 = self.ConvLayer1Dim(inp, filters, kernelSize)
		conv2 = self.ConvLayer1Dim(conv1, filters, kernelSize)
		out = layers.Add()([inp, conv2])

		if newDepth > -1:
			pool = layers.MaxPooling2D((2, 2))(out)
			return self.ConvLayer1Dim(pool, newDepth, 1)

		return out

	def build_model(self, shape, numClasses, weights = ''):

		inputLayer = tf.keras.Input(shape=shape, name='simple_inp_layer')

		layersCount = 96

		layer = self.ConvLayer1Dim(inputLayer, layersCount)

		for _ in range(3):
			layer = self.ResidualLayer2(layer, layersCount, 3,  newDepth = -1)
			layer = self.ResidualLayer2(layer, layersCount, 3, newDepth = -1)
			layer = self.ResidualLayer2(layer, layersCount, 3, newDepth = layersCount * 2)
			layersCount = layersCount * 2

		flatten = layers.Flatten()(layer)
		dense = layers.Dense(32, activation=tf.nn.relu)(flatten)
		dense = layers.Dense(32, activation=tf.nn.relu)(dense)
		dense = layers.Dense(32, activation=tf.nn.relu)(dense)
		out = layers.Dense(numClasses, activation=tf.nn.sigmoid)(dense)

		return tf.keras.models.Model(inputs=inputLayer, outputs=out)