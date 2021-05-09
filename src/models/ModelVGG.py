import tensorflow as tf
from tensorflow.keras import layers

from src.models.base.BaseModel import BaseModel

class ModelVGG(BaseModel):

	def get_name(self):
		return "vgg"

	def build_model(self, shape, numClasses, weights = 'imagenet'):

		inputLayer = tf.keras.Input(shape=shape, name='simple_inp_layer')

		includeTop = tf.compat.v1.constant(False, tf.bool)
		net = tf.keras.applications.VGG16(input_tensor=inputLayer, include_top=includeTop, weights=weights)
		net.trainable = False

		flatten = layers.Flatten()(net.output)
		dense = layers.Dense(256, activation=tf.nn.relu)(flatten)
		dense1 = layers.Dense(128, activation=tf.nn.relu)(dense)
		dense2 = layers.Dense(64, activation=tf.nn.relu)(dense1)
		dense3 = layers.Dense(32, activation=tf.nn.relu)(dense2)
		out = layers.Dense(numClasses, activation=tf.nn.sigmoid)(dense3)

		return tf.keras.models.Model(inputs=inputLayer, outputs=out)