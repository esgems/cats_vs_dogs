import tensorflow as tf
from tensorflow.keras import layers

from src.models.base.BaseModel import BaseModel

class ModelResnet50(BaseModel):

	def get_name(self):
		return "resnet50"

	def build_model(self, shape, numClasses, weights = 'imagenet'):

		inputLayer = tf.keras.Input(shape=shape, name='simple_inp_layer')

		includeTop = tf.compat.v1.constant(False, tf.bool)
		net = tf.keras.applications.ResNet50(input_tensor=inputLayer, include_top=includeTop, weights=weights)
		net.trainable = False

		flatten = layers.Flatten()(net.output)
		dense = layers.Dense(64, activation=tf.nn.relu)(flatten)
		dense = layers.Dense(64, activation=tf.nn.relu)(dense)
		dense = layers.Dense(64, activation=tf.nn.relu)(dense)
		out = layers.Dense(numClasses, activation=tf.nn.sigmoid)(dense)

		return tf.keras.models.Model(inputs=inputLayer, outputs=out)