import os

import tensorflow_cloud as tfc

from src.dataset_manager.TFRecordLoader import TFRecordLoader
from src.models.ModelCore import ModelCore
from src.metrics.Metrics import Metrics

MODE = 0 # 0 - train, 1 - accuracytest 2- test image

#model
MODEL_TYPE = 1

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

LOCAL_DATA_PATH = "D:/Projects/test_rab_2_local_data"
LOCAL_MODEL_PATH = os.path.join(LOCAL_DATA_PATH, "saved_models", MODEL_NAME, MODEL_NAME)

if MODE == 0:
	GCP_BUCKET = "projectcatsbucket"

	BATCH_SZE = 16
	TRAIN_EPOCHS = 60
	TRAIN_EPOCHS_STEPS = 50
	VAL_EPOCHS = 10
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
		
		import cv2
		import tensorflow as tf
		from src.viz.Viz import Viz

		viz = Viz()

		folder = "D:\\Projects\\test_rab_2_local_data\\test_images\\"

		for filename in os.listdir(folder):
			image = cv2.imread(os.path.join(folder,filename))
			image = tf.image.resize(image, IM_SIZE[:2])
			image = image / 255.0
			image = tf.expand_dims(image, axis=0)

			pre = model.predict(image)

			predLabels, predBbxs = tf.split(pre, (1, 4), axis=-1)
			predBbxs = predBbxs * tf.cast(IM_SIZE[0], tf.float32)
			predBbxs = predBbxs.numpy().astype(int)

			print(predLabels[0].numpy())
			
			viz.show1(image[0].numpy(), predBbxs[0])

else:
	print("Remote testing is not available.")