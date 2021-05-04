import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.compat.v1.enable_eager_execution()

class ModelVGG():

	def DeconvLayer(self, x1, x2, filters, kernel_size=4, strides=(2, 2), padding='same', activation=tf.nn.relu):
		result = layers.Conv2DTranspose(filters=filters, kernel_size=4, strides=(2, 2),
			kernel_initializer= tf.compat.v1.random_normal_initializer(stddev=0.01),
			kernel_regularizer= tf.keras.regularizers.l2(1e-4),
			activation=activation,
			padding=padding)(x1)

		if x2 is None:
			return result

		return tf.add(result, x2) 

	def BuildModel(self, inp, numClasses):

		block1_pool = 'block1_pool'
		block2_pool = 'block2_pool'
		block3_pool = 'block3_pool'
		block4_pool = 'block4_pool'
		block5_pool = 'block5_pool'

		includeTop = tf.compat.v1.constant(False, tf.bool)
		vgg16 = tf.keras.applications.VGG16(input_tensor=inp, include_top=includeTop, weights='imagenet')
		vgg16.trainable = False

		# layer1 = vgg16.get_layer(block1_pool)
		# layer2 = vgg16.get_layer(block2_pool)
		# layer3 = vgg16.get_layer(block3_pool)
		# layer4 = vgg16.get_layer(block4_pool)
		layer5 = vgg16.get_layer(block5_pool)
		
		# l5_logit = layers.Conv2D(filters=400, kernel_size=1, activation=tf.nn.relu, padding='same', name='logit5_conv')(layer5.output)
		# l4_logit = layers.Conv2D(filters=200, kernel_size=1, activation=tf.nn.relu, padding='same', name='logit4_conv')(layer4.output)
		# l3_logit = layers.Conv2D(filters=100, kernel_size=1, activation=tf.nn.relu, padding='same', name='logit3_conv')(layer3.output)
		# l2_logit = layers.Conv2D(filters=50, kernel_size=1, activation=tf.nn.relu, padding='same', name='logit2_conv')(layer2.output)
		# l1_logit = layers.Conv2D(filters=25, kernel_size=1, activation=tf.nn.relu, padding='same', name='logit1_conv')(layer1.output)
		
		# deconv1 = self.DeconvLayer(l5_logit, l4_logit, filters=200)
		# deconv2 = self.DeconvLayer(deconv1, l3_logit, filters=100)
		# deconv3 = self.DeconvLayer(deconv2, l2_logit, filters=50)
		# deconv4 = self.DeconvLayer(deconv3, l1_logit, filters=25)
		flatten = layers.Flatten()(layer5.output)
		dense = layers.Dense(512, activation=tf.nn.relu)(flatten)
		dense2 = layers.Dense(256, activation=tf.nn.relu)(dense)
		dense3 = layers.Dense(128, activation=tf.nn.relu)(dense2)
		out = layers.Dense(numClasses, activation=None)(dense3)

		initial_learning_rate = 0.0001
		optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

		model = keras.models.Model(inputs=inp, outputs=out)

		model.compile(loss=self.Loss(), optimizer=optimizer)#, metrics=[self.MetricIOU()])

		return model

	def Loss(self):
		def NewLoss(y_true, y_pred):
			
			ob1, bbx = tf.split(y_pred, (1, 4), axis=-1)
			ob1 = tf.reshape(ob1, [-1])
			objectness = tf.sigmoid(ob1)
			l1 = tf.keras.losses.binary_crossentropy(y_true[:,0], objectness)

			diff = y_true[:,1:] - bbx
			summ = tf.reduce_sum(tf.square(diff), axis=-1)
			summ2 = tf.reduce_sum(summ)

			return l1 + summ2

		return NewLoss


	def MetricIOU(self, smooth=1e-5):
		def iou_score(y_true, y_pred):

			# print("---------------------------------------------")
			# print(y_true.shape, " => ", y_pred.shape)
			# print("---------------------------------------------")

			axes = (0, 1, 2)
			y_pred =  tf.keras.backend.cast(y_pred,  tf.keras.backend.floatx())
			intersection =  tf.keras.backend.sum(y_true[1:] * y_pred[1:], axis=axes)
			union =  tf.keras.backend.sum(y_true[1:] + y_pred[1:], axis=axes) - intersection

			score = (intersection + smooth) / (union + smooth)
			score =  tf.keras.backend.mean(score)

			return score

		return iou_score
