import time
import numpy as np
import tensorflow as tf

THRESHOLD = 0.5

class Metrics():

	def __init__(self):
		self.mAcc = 0.0
		self.mIoU = 0.0
		self.mSpeed = 0.0

	@staticmethod
	def AccuracyIoU(y_true, y_pred):

		# intersecion / (area1 + area2 - intersecion)

		_, y_true = tf.split(y_true, (1, 4), axis=-1)
		_, y_pred = tf.split(y_pred, (1, 4), axis=-1)

		intersection_width = tf.maximum(tf.minimum(y_true[..., 2], y_pred[..., 2]) - tf.maximum(y_true[..., 0], y_pred[..., 0]), 0)
		intersection_height = tf.maximum(tf.minimum(y_true[..., 3], y_pred[..., 3]) - tf.maximum(y_true[..., 1], y_pred[..., 1]), 0)
		intersection_area = intersection_width * intersection_height
		true_area = (y_true[..., 2] - y_true[..., 0]) * (y_true[..., 3] - y_true[..., 1])
		pred_area = (y_pred[..., 2] - y_pred[..., 0]) * (y_pred[..., 3] - y_pred[..., 1])

		return tf.reduce_sum(intersection_area / (true_area + pred_area - intersection_area)) / tf.cast(tf.shape(y_true)[0], tf.float32)

	@staticmethod
	def AccuracyClassification(y_true, y_pred):

		y_true, _ = tf.split(y_true, (1, 4), axis=-1)
		y_pred, _ = tf.split(y_pred, (1, 4), axis=-1)

		preds = tf.cast((y_pred > THRESHOLD), tf.float32)
		diffs = 1.0 - abs(y_true - preds)
		summ = tf.cast(tf.reduce_sum(diffs), tf.float32)
		sub = tf.cast(tf.shape(y_true)[0], tf.float32)
		return summ / sub

	def calc(self, model, data):

		sumIoU = 0
		sumAcc = 0
		sumSpeed = 0
		count = 0

		for raw_record in data.as_numpy_iterator():
			img, gtLabels = raw_record

			time1 = time.time()

			result = model.predict(img)
			
			time2 = time.time()
			speed = (time2 - time1) / np.shape(gtLabels)[0]
			sumSpeed += speed

			gtLabels = tf.cast(gtLabels, tf.float32)
			sumIoU += Metrics.AccuracyIoU(gtLabels, result)
			sumAcc += Metrics.AccuracyClassification(gtLabels, result)
			count += 1
			# print("gt: ", gtLabels.numpy()[0, 0], ", res: ", result[0, 0], ", acc: ", Metrics.AccuracyClassification(gtLabels, result).numpy())

		self.mIoU = sumIoU / count
		self.mAcc = sumAcc / count
		self.mSpeed = sumSpeed / count

	def print(self):
		print("mIoU: ", self.mIoU)
		print("mAcc: ", self.mAcc)
		print("mSpeed: ", self.mSpeed)

	def save(self, path, trainSize, valSize):
		res = "mIoU: {}%, classification accuracy: {}%, inf. time: {}, train: {}, valid: {} " \
			.format(int(self.mIoU * 100), int(self.mAcc * 100), self.mSpeed, trainSize, valSize)

		with open(path, 'w') as f:
			f.write(res)