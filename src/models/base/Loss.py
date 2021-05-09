import tensorflow as tf

class Loss():
	@staticmethod
	def LossBinary(y_true, y_pred):
		return tf.keras.losses.binary_crossentropy(y_true, y_pred)

	@staticmethod
	def LossIOU(y_true, y_pred):

		obTrue, bbxTrue = tf.split(y_true, (1, 4), axis=-1)
		obPred, bbxPred = tf.split(y_pred, (1, 4), axis=-1)

		err1 = tf.keras.losses.binary_crossentropy(obTrue, obPred)
		err2 = tf.keras.losses.mean_squared_error(bbxTrue, bbxPred)

		return err1 + err2