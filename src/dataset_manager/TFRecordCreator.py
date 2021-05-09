import tensorflow as tf 

import numpy as np
import glob

class TFRecordCreator():
	def to_int64_feature(self, value):
		return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

	def to_bytes_feature(self, value):
		return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

	def create_from_txt_files(self, folder, outFile):
		labels = glob.glob(folder + "*.txt")

		writer = tf.io.TFRecordWriter(outFile)

		for label in labels:
			file1 = open(label, 'r')

			for line in file1.readlines():
				bboxData = np.array(line.split(" ")).astype(np.int32)
				path = label.replace(".txt", ".jpg")

				with tf.io.gfile.GFile(path, 'rb') as fid:
					img = fid.read()

					feature = { 'label': self.to_int64_feature(bboxData[0]), 
								'xmin': self.to_int64_feature(bboxData[1]), 
								'ymin': self.to_int64_feature(bboxData[2]), 
								'xmax': self.to_int64_feature(bboxData[3]), 
								'ymax': self.to_int64_feature(bboxData[4]), 
								'image': self.to_bytes_feature(img)}

					example = tf.train.Example(features=tf.train.Features(feature=feature))

					writer.write(example.SerializeToString())

		writer.close()