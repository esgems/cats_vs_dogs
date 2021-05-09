import os

import tensorflow as tf
import tensorflow_cloud as tfc

from src.dataset_manager.TFRecordLoader import TFRecordLoader
from src.models.ModelCore import ModelCore
from src.metrics.Metrics import Metrics

MODE = 0 		# 0 - train, 1 - accuracytest 2- test image
MODEL_TYPE = 2 	# 0 - vgg, 1 - resnet, 2 - custom

if MODEL_TYPE == 0:
	MODEL_NAME = "modelVgg"
elif MODEL_TYPE == 1:
	MODEL_NAME = "modelResNet"
else:
	MODEL_NAME = "modelOwn"

# size
IM_SIZE = (256, 256, 3)
OUT_SIZE = 5 # [class, [x1, y1], [x2, y2]]
LEARNING_RATE = 0.0001

loader = TFRecordLoader(IM_SIZE)

LOCAL_DATA_PATH = "D:/Projects/cats_vs_dogs_data"
LOCAL_MODEL_PATH = os.path.join(LOCAL_DATA_PATH, "saved_models", MODEL_NAME, MODEL_NAME)

if MODE == 0:
	GCP_BUCKET = "projectcatsbucket"

	BATCH_SZE = 16
	TRAIN_EPOCHS = 80
	TRAIN_EPOCHS_STEPS = 80
	VAL_EPOCHS = 15
	SAVE_FREQ = 20
	
	data_directory = os.path.join(LOCAL_DATA_PATH, "data/train.tfrecords")
	checkpoint_path = os.path.join(LOCAL_MODEL_PATH, "checkpoints")
	save_model_path = os.path.join(LOCAL_MODEL_PATH)

	if tfc.remote():
		data_directory = os.path.join("gs://", GCP_BUCKET, "train.tfrecords")

		REMOTE_MODEL_PATH = os.path.join("gs://", GCP_BUCKET, "saved_models", MODEL_NAME)
		checkpoint_path = os.path.join(REMOTE_MODEL_PATH, "checkpoints")
		save_model_path = os.path.join(REMOTE_MODEL_PATH, MODEL_NAME)
	
	train, test = loader.load_train(data_directory, BATCH_SZE)

	model = ModelCore(MODEL_TYPE, IM_SIZE, OUT_SIZE, LEARNING_RATE)
	model.train(train, test, checkpoint_path, TRAIN_EPOCHS, TRAIN_EPOCHS_STEPS, VAL_EPOCHS, SAVE_FREQ)
	model.save_weights(save_model_path)

elif not tfc.remote():

	model = ModelCore(MODEL_TYPE, IM_SIZE, OUT_SIZE, 0)
	model = model.load_model(LOCAL_MODEL_PATH)

	if MODE == 1:
		data_directory = os.path.join(LOCAL_DATA_PATH, "data", "val.tfrecords")

		test = loader.load_test(data_directory)

		metrics = Metrics()
		metrics.calc(model, test)
		metrics.print()
		
		data_directory = os.path.join(LOCAL_DATA_PATH, "data/train.tfrecords")
		train, test = loader.load_train(data_directory, 0)
		trainSize = sum(1 for _ in train)
		testSize = sum(1 for _ in test)
		metrics.save("result.txt", trainSize, testSize)
		
	else:
		
		from src.dataset_manager.LocalDataLoader import LocalDataLoader
		from src.viz.Viz import Viz

		IMAGES_FOLDER = os.path.join(LOCAL_DATA_PATH, "test_images")

		for filename in os.listdir(IMAGES_FOLDER):
			image = LocalDataLoader.load_image(os.path.join(IMAGES_FOLDER, filename), IM_SIZE[:2])
			image = tf.expand_dims(image, axis=0)

			pre = model.predict(image)

			predLabels, predBbxs = tf.split(pre, (1, 4), axis=-1)
			predBbxs = predBbxs * tf.cast(IM_SIZE[0], tf.float32)
			predBbxs = predBbxs.numpy().astype(int)

			name = "cat" if predLabels[0].numpy()[0] < 0.5  else "dog"
			Viz.show1(image[0].numpy(), predBbxs[0], name)

else:
	print("Remote testing is not available.")