import tensorflow as tf 

class TFRecordLoader():

	def __init__(self, im_size):
		self.im_size = im_size

		self.feature_description = {
			'label': tf.io.FixedLenFeature([], tf.int64),
			'xmin': tf.io.FixedLenFeature([], tf.int64),
			'ymin': tf.io.FixedLenFeature([], tf.int64),
			'xmax': tf.io.FixedLenFeature([], tf.int64),
			'ymax': tf.io.FixedLenFeature([], tf.int64),
			'image': tf.io.FixedLenFeature([], tf.string),
		}

	def parse_tfrecord(self, tfrecord):
		x = tf.io.parse_single_example(tfrecord, self.feature_description)
		x_train = tf.image.decode_jpeg(x['image'], channels=3)

		imageShape = tf.shape(x_train)
		
		scaleX = 1.0 / tf.cast(imageShape[1], tf.float64)
		scaleY = 1.0 / tf.cast(imageShape[0], tf.float64)

		label = tf.cast(x['label'], tf.float64)
		p1x = tf.multiply(tf.cast(x['xmin'], tf.float64), scaleX)
		p1y = tf.multiply(tf.cast(x['ymin'], tf.float64), scaleY)
		p2x = tf.multiply(tf.cast(x['xmax'], tf.float64), scaleX)
		p2y = tf.multiply(tf.cast(x['ymax'], tf.float64), scaleY)
		
		y_train = tf.stack([label , p1x, p1y, p2x, p2y ])

		return x_train, y_train

	def getDataset(self, path):
		files = tf.data.Dataset.list_files(path)
		dataset = files.flat_map(tf.data.TFRecordDataset)
		return dataset.map(lambda x: self.parse_tfrecord(x))


	def transform_images(self, x_train, size, aug = True):
		if aug:
			x_train = tf.image.random_hue(x_train, 0.04)
			x_train = tf.image.random_brightness(x_train, 0.015)
			x_train = tf.image.random_contrast(x_train, 0.8, 1.1)

		x_train = tf.image.resize(x_train, size)
		x_train = x_train / 255.0
		return x_train

	def load_train(self, path, batch_sze = 24, shuffle = True):
		# load data
		dataset = self.getDataset(path)

		if shuffle:
			dataset = dataset.shuffle(buffer_size=512, reshuffle_each_iteration=False)

		# train-test split
		is_test = lambda x, y: x % 5 == 0
		is_train = lambda x, y: not is_test(x, y)
		recover = lambda x, y: y

		train = dataset.enumerate().filter(is_train).map(recover)
		train = train.map(lambda x, y: ((self.transform_images(x, self.im_size[:2], True)), y))
		if batch_sze > 0:
			train = train.batch(batch_sze).repeat()
		train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

		test = dataset.enumerate().filter(is_test).map(recover)
		test = test.map(lambda x, y: ((self.transform_images(x, self.im_size[:2], False)), y))
		if batch_sze > 0:
			test = test.batch(batch_sze).repeat()
		test = test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
			
		return train, test
	
	def load_test(self, path, batch_sze = 1):
		dataset = self.getDataset(path)
		dataset = dataset.map(lambda x, y: ((self.transform_images(x, self.im_size[:2], False)), y))

		dataset = dataset.batch(batch_sze)
		dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

		return dataset