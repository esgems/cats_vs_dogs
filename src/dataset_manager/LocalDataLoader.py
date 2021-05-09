import cv2
import tensorflow as tf 

class LocalDataLoader():
	@staticmethod
	def load_image(path, size):
		image = cv2.imread(path)
		image = tf.image.resize(image, size)
		image = image / 255.0
		return image