import tensorflow as tf 

import numpy as np
import glob

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

folder = './data/valid/'
labels = glob.glob(folder + "*.txt")
bounds = [[], []]
paths = [[], []]

for label in labels:
	file1 = open(label, 'r')
	for line in file1.readlines():
		ll = np.array(line.split(" ")).astype(np.int32)
		bounds[ll[0] - 1].append(np.array(ll[1:]))
		paths[ll[0] - 1].append(label.replace(".txt", ".jpg"))

tfrecord_filename = 'something1_val.tfrecords'
writer = tf.io.TFRecordWriter(tfrecord_filename)

i = 0
for bound in bounds:
	j = 0
	for bb in bound:

		with tf.io.gfile.GFile(paths[i][j], 'rb') as fid:
			img = fid.read()

			label = i

			feature = { 'label': _int64_feature(label), 
						'xmin': _int64_feature(bb[0]), 
						'ymin': _int64_feature(bb[1]), 
						'xmax': _int64_feature(bb[2]), 
						'ymax': _int64_feature(bb[3]), 
						'image': _bytes_feature(img)}

			example = tf.train.Example(features=tf.train.Features(feature=feature))

			writer.write(example.SerializeToString())

		j += 1

	i += 1

writer.close()