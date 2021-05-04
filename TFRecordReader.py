import tensorflow as tf 
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)

import cv2
import numpy as np

from ModelVGG import ModelVGG

feature_description = {
    'label': tf.io.FixedLenFeature([], tf.int64),
    'xmin': tf.io.FixedLenFeature([], tf.int64),
    'ymin': tf.io.FixedLenFeature([], tf.int64),
    'xmax': tf.io.FixedLenFeature([], tf.int64),
    'ymax': tf.io.FixedLenFeature([], tf.int64),
    'image': tf.io.FixedLenFeature([], tf.string),
}

# dataset = tf.data.TFRecordDataset(["something1.tfrecords"])

def parse_tfrecord(tfrecord):
	x = tf.io.parse_single_example(tfrecord, feature_description)
	x_train = tf.image.decode_jpeg(x['image'], channels=3)

	ss = tf.shape(x_train)
	scaleX = tf.cast(ss[1], tf.float64) / 213.0
	scaleY = tf.cast(ss[0], tf.float64) / 213.0

	ssx = 1.0 / scaleX
	ssy = 1.0 / scaleY

	p1x = tf.multiply(tf.cast(x['xmin'], tf.float64), ssx)
	p1y = tf.multiply(tf.cast(x['ymin'], tf.float64), ssy)
	p2x = tf.multiply(tf.cast(x['xmax'], tf.float64), ssx)
	p2y = tf.multiply(tf.cast(x['ymax'], tf.float64), ssy)

	print(scaleX)
	print(scaleY)
	
	y_train = tf.stack([ tf.cast(x['label'], tf.float64), p1x, p1y, p2x, p2y ])

	return x_train, y_train

def getDataset(validate):
	files = tf.data.Dataset.list_files("something1_val.tfrecords" if validate else "something1.tfrecords")
	dataset = files.flat_map(tf.data.TFRecordDataset)
	return dataset.map(lambda x: parse_tfrecord(x))

def transform_images(x_train, size):
	x_train = tf.image.resize(x_train, (size, size))
	x_train = x_train / 255.0
	return x_train

def transform_labels(y_train):
	return y_train

def transform_data(x, size):
	return x
	
def createModel():
	
	shape = (213, 213)
	inputs = tf.keras.Input(shape=(shape[0], shape[1], 3), name='digits')

	model = ModelVGG()
	
	return model.BuildModel(inputs, 5)

def train():
	full_dataset = getDataset(False)
	full_dataset = full_dataset.shuffle(buffer_size=512)  # TODO: not 1024
	# full_dataset = full_dataset.map(lambda x, y: (transform_images(x, 213), transform_labels(y)))
	full_dataset = full_dataset.map(lambda x, y: ((transform_images(x, 213)), y))

	train_size = 2900
	train_dataset = full_dataset.take(train_size)
	train_dataset = train_dataset.batch(12).repeat()
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	val_full_dataset = getDataset(True)
	val_dataset = val_full_dataset.take(500)
	val_dataset = val_dataset.batch(12).repeat()
	val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

	# for raw_record in full_dataset.as_numpy_iterator():
	# 	img, data = raw_record
	# 	data2 = (data * 1.0).astype(np.int32)
	# 	cv2.rectangle(img, (data2[1], data2[2]), (data2[3], data2[4]), (255, 0, 0), 5)
	# 	cv2.imshow("img", img)
	# 	cv2.waitKey()

	callbacks = [
		# ReduceLROnPlateau(verbose=0),
		# EarlyStopping(patience=3, verbose=0),
		ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf', verbose=1, period=1, save_weights_only=True),
		TensorBoard(log_dir='logs')
		# MyCallback()
	]

	model = createModel()
	# print(model.summary())
	history = model.fit(train_dataset, epochs=40, callbacks=callbacks, validation_data=val_dataset, steps_per_epoch=80, validation_steps=5)

def test():
	model = createModel()
	epoch = 6
	model.load_weights("./checkpoints/yolov3_train_22.tf")
	
	full_dataset = getDataset(False)
	full_dataset = full_dataset.shuffle(buffer_size=512)  # TODO: not 1024
	full_dataset = full_dataset.map(lambda x, y: ((transform_images(x, 213)), y))

	train_dataset = full_dataset.take(10)
	train_dataset = train_dataset.batch(4).repeat()
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)


	for raw_record in full_dataset.as_numpy_iterator():
		img, data = raw_record

		if data[0] == 0:
			continue

		sh = tf.shape(img)
		im2 = np.reshape(img, (-1, sh[0], sh[1], sh[2]))
		pre = model.predict(im2)
		print(pre)
		data2 = (pre[0] * 1.0).astype(np.int32)
		cv2.rectangle(img, (data2[1], data2[2]), (data2[3], data2[4]), (255, 0, 0), 5)
		cv2.imshow("img", img)
		cv2.waitKey()

test()
# train()

