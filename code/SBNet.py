import tensorflow as tf
import numpy as np
import os
import time
import pickle
import tarfile


class Tracer:
	def __init__(self, log_root_dir: str, saver_root_dir: str, model_name: str, save_count: int):
		self.model_name = model_name
		self.saver_dir = saver_root_dir + self.model_name
		self.log_dir = log_root_dir + self.model_name
		self.process_dir = self.saver_dir + "/process"
		self.best_dir = self.saver_dir + "/best"
		if not os.path.exists(self.process_dir):
			os.makedirs(self.process_dir)
		if not os.path.exists(self.process_dir):
			os.makedirs(self.process_dir)
		if not os.path.exists(self.best_dir):
			os.makedirs(self.best_dir)
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
		self.process_saver_file = self.process_dir + '/saver.dat'
		self.best_saver_file = self.best_dir + "/saver.dat"
		log_file = self.log_dir + "/log.txt"
		statistic_file = self.log_dir + "/statistic.txt"
		csv_file = self.log_dir + "/log_csv.csv"
		self.log = open(log_file, "a")
		need_title = not os.path.exists(csv_file)
		self.statistic = open(statistic_file, "a+")
		self.csv = open(csv_file, "a+")
		if need_title:
			self.csv.write("accuracy,loss,t-loss\n")
			self.csv.flush()
		self.process_saver = None
		self.best_saver = None
		self.total_second = 0
		self.best_acc = 0.0
		self.best_loss = 0.0
		self.save_time = 0.0
		self.count = 0
		self.save_count = save_count

	def start_time_trace(self):
		self.save_time = time.time()

	def inc_count(self):
		self.count += 1
		if (self.count % self.save_count) == (self.save_count - 1):
			return True
		return False

	def load(self, sess, best=False):
		self.process_saver = tf.train.Saver(max_to_keep=1)
		self.best_saver = tf.train.Saver(max_to_keep=1)
		load = os.path.exists(self.process_dir + "/checkpoint")
		best_load = os.path.exists(self.best_dir + "/checkpoint")
		self.statistic.seek(0)
		sta_data = self.statistic.readline()
		if len(sta_data) == 0:
			self.total_second = 0.0
			self.best_acc = 0.0
			self.best_loss = 0.0
		else:
			detail_sta_data = sta_data.split(",")
			self.total_second = float(detail_sta_data[0])
			self.best_acc = float(detail_sta_data[1])
			self.best_loss = float(detail_sta_data[2])
		if best:
			if best_load:
				self.best_saver.restore(sess, tf.train.latest_checkpoint(self.best_dir))
				print("Load from best. Best acc:%.5f loss:%.5f" % (self.best_acc, self.best_loss))
			else:
				print("Not exist best checkpoint")
				raise RuntimeError()
		elif load:
			self.process_saver.restore(sess, tf.train.latest_checkpoint(self.process_dir))
			print("Load model done. Best acc:%.5f loss:%.5f" % (self.best_acc, self.best_loss))
		else:
			self.log.truncate(0)
			self.log.seek(0)
			self.csv.truncate(0)
			self.csv.seek(0)
			self.csv.write("accuracy,loss,t-loss\n")
			self.csv.flush()
			self.total_second = 0.0
			self.best_acc = 0.0
			self.best_loss = 99999
			print("Need not load model")
		self.log.write(time.asctime(time.localtime(time.time())) + "\n")
		self.log.flush()
		return load or (best and best_load)

	def save(self, sess, acc: float, loss: float, t_loss: float, t_acc: float, append_info=""):
		info = "Saving "
		current_second = time.time()
		used_time = current_second - self.save_time
		self.process_saver.save(sess, self.process_saver_file)
		self.total_second += used_time
		self.save_time = current_second
		if self.is_best(acc, loss):
			info = info + "Best "
			self.best_saver.save(sess, self.best_saver_file)
			self.best_acc = acc
			self.best_loss = loss
		else:
			info = info + "     "
		self.statistic.seek(0)
		self.statistic.truncate(0)
		self.statistic.write(str(self.total_second) + "," + str(self.best_acc) + "," + str(self.best_loss))
		self.statistic.flush()
		info = info + "time: %f acc: %.5f loss: %.5f t-loss: %.5f t-acc: %.5f %s" % (
			used_time, acc, loss, t_loss, t_acc, append_info)
		print(info)
		self.log.write(info + "\n")
		self.log.flush()
		self.csv.write("%.5f,%.5f,%.5f\n" % (acc, loss, t_loss))
		self.csv.flush()

	def is_best(self, acc, loss):
		return acc > 0 and (acc > self.best_acc or ((self.best_acc / self.best_loss) < (acc / loss)))

	def print_log(self, info: str):
		if not info.endswith("\n"):
			info = info + "\n"
		self.log.write(info)
		self.log.flush()

	def close(self):
		if self.statistic is not None:
			self.statistic.close()
			self.statistic = None
		if self.log is not None:
			self.log.close()
			self.log = None
		if self.csv is not None:
			self.csv.close()
			self.csv = None

	def __del__(self):
		self.close()


class Utils:
	@staticmethod
	def label_to_one_hot(labels, hot_size):
		result = []
		for index in labels:
			temp = [0] * hot_size
			temp[index] = 1
			result.append(temp)
		return np.array(result, dtype=np.float32)

	@staticmethod
	def read_cifar10_data():
		def reshape_image(image):
			result = []
			data = np.reshape(image, [-1, 32, 32])
			for i in range(data.shape[0] // 3):
				temp = np.dstack(data[3 * i:3 * i + 3])
				result.append(temp[:])
			return np.array(result)

		ext_dir = "res/ext"
		zip_file = "res/cifar-10-python.tar.gz"
		data_dir = ext_dir + "/cifar-10-batches-py/"
		data_file = data_dir + "data_batch_"
		train_file = data_dir + "test_batch"
		image_data = np.empty(shape=[0, 32 * 32 * 3])
		image_label = np.empty(shape=[0, 10])
		if not os.path.exists(ext_dir):
			tarfile.open(zip_file, "r:gz").extractall(ext_dir)
		for i in range(1, 6):
			with open(data_file + str(i), 'rb') as fo:
				dict = pickle.load(fo, encoding='bytes')
				image_data = np.concatenate((image_data, dict[b'data']))
				image_label = np.concatenate((image_label, Utils.label_to_one_hot(dict[b'labels'], 10)))
		with open(train_file, 'rb') as fo:
			dict = pickle.load(fo, encoding='bytes')
			test_data = dict[b'data']
			test_label = Utils.label_to_one_hot(dict[b'labels'], 10)
		image_data = reshape_image(image_data)
		test_data = reshape_image(test_data)
		return image_data, image_label, test_data, test_label

	@staticmethod
	def read_iris_data():
		from sklearn import datasets
		iris = datasets.load_iris()
		iris_data = iris.data
		iris_label = iris.target
		from sklearn.model_selection import train_test_split
		train_data, test_data, train_label, test_label = train_test_split(iris_data, iris_label, test_size=0.3,
			random_state=0)
		train_data = np.reshape(train_data, [len(train_data), 1, 1, 4])
		train_data = np.concatenate([train_data, np.reshape(np.sum(train_data, 3), [-1, 1, 1, 1])], 3)
		test_data = np.reshape(test_data, [len(test_data), 1, 1, 4])
		test_data = np.concatenate([test_data, np.reshape(np.sum(test_data, 3), [-1, 1, 1, 1])], 3)
		train_label = Utils.label_to_one_hot(train_label, 3)
		test_label = Utils.label_to_one_hot(test_label, 3)
		return train_data, train_label, test_data, test_label


class BaseNet:
	def __init__(self, input_shape, label_size: int, batch_count: int, lr: float):
		"""
		init all variables
		:param input_shape: format: [None,width,height,channels], for example: [None,32,32,3]
		:param label_size: how many labels
		:param batch_count: how many pieces of data in every batch
		:param lr: learning rate
		"""
		self.batch_count = batch_count
		self.input_shape = input_shape
		self.label_size = label_size
		self.training = tf.placeholder(dtype=tf.bool)
		self.input = tf.placeholder(dtype=tf.float32, shape=self.input_shape, name="input")
		self.label = tf.placeholder(dtype=tf.float32, shape=[None, self.label_size], name="label")
		self.wather1 = tf.Variable(tf.constant(0, dtype=tf.int32), False)
		self.wather2 = tf.Variable(tf.constant(0, dtype=tf.int32), False)
		self.distance_gpu = []
		self.lr = lr
		self.loss = 0.0
		self.accuracy = 0.0
		self.optimizer = None
		self.prediction = None
		self.sb_argv_list = []
		self.sb_storage_box = []
		self.distance_gpu = []
		self.sb_init_map = []
		self.storage_vector_data_holder = []
		self.input_feature_vector_data_holder = []
		self.predict_result_gpu = []
		self.mse_result_gpu = []
		self.wl_predict = []
		self.wl_distance = []
		self.wl_mse = []
		self.variables = {}
		self.training_count = 0

	def construct_model(self):
		pass

	def calculate_loss(self, sess, train_data, train_label, test_data, test_label):
		if len(self.sb_argv_list) != 0:
			storage_vector_data = [[]] * len(self.sb_argv_list)
			for argv_index in range(len(self.sb_argv_list)):
				input_layer, storage_size = self.sb_argv_list[argv_index]
				per_size = storage_size * 2
				input_feature_vector_data = sess.run(input_layer,
					feed_dict={self.input: test_data, self.training: False})
				input_feature_vector_data = np.reshape(input_feature_vector_data, [len(input_feature_vector_data), -1])
				s_data = sess.run(input_layer, feed_dict={self.input: np.reshape(self.sb_storage_box[argv_index],
					[-1, self.input_shape[1], self.input_shape[2], self.input_shape[3]]), self.training: False})
				storage_vector_data[argv_index] = np.reshape(s_data, [self.label_size, per_size, -1])
				sess.run(self.wl_predict[argv_index],
					feed_dict={self.storage_vector_data_holder[argv_index]: storage_vector_data[argv_index],
						self.input_feature_vector_data_holder[argv_index]: input_feature_vector_data})
			accuracy, loss, predict = sess.run((self.accuracy, self.loss, self.prediction),
				feed_dict={self.input: test_data, self.label: test_label})
			for argv_index in range(len(self.sb_argv_list)):
				input_layer, storage_size = self.sb_argv_list[argv_index]
				input_feature_vector_data = sess.run(input_layer,
					feed_dict={self.input: train_data, self.training: False})
				input_feature_vector_data = np.reshape(input_feature_vector_data, [len(input_feature_vector_data), -1])
				sess.run(self.wl_predict[argv_index],
					feed_dict={self.storage_vector_data_holder[argv_index]: storage_vector_data[argv_index],
						self.input_feature_vector_data_holder[argv_index]: input_feature_vector_data})
			train_loss, train_acc = sess.run((self.loss, self.accuracy),
				feed_dict={self.input: train_data, self.label: train_label})
		else:
			accuracy, loss, predict = sess.run((self.accuracy, self.loss, self.prediction),
				feed_dict={self.input: test_data, self.label: test_label})
			train_loss, train_acc = sess.run((self.loss, self.accuracy),
				feed_dict={self.input: train_data, self.label: train_label})
		return accuracy, loss, train_loss, train_acc, predict

	def train(self, sess, train_input, train_label):
		if len(self.sb_argv_list) != 0:
			start_time = time.time()
			argv_list = self.sb_argv_list
			self.sb_result = [[]] * self.batch_count
			for argv_index in range(len(argv_list)):
				argv = argv_list[argv_index]
				input_layer, storage_size = argv[0:2]
				per_size = (storage_size) * 2
				sb_init_map = self.sb_init_map[argv_index]
				storage = self.sb_storage_box[argv_index]
				storage_data_result = sess.run(input_layer, feed_dict={self.input: np.reshape(storage,
					[-1, self.input_shape[1], self.input_shape[2], self.input_shape[3]]), self.training: False})
				# [label_size, per_size, layer.shape[1]*[2]*[3]
				storage_vector_data = np.reshape(storage_data_result, [self.label_size, per_size, -1])
				result = sess.run(self.wl_distance[argv_index],
					feed_dict={self.storage_vector_data_holder[argv_index]: storage_vector_data})
				# shape: [label,ps,ps]
				distance = sess.run(self.distance_gpu[argv_index])

				input_feature_vector_data = sess.run(input_layer,
					feed_dict={self.input: train_input, self.training: False})
				input_feature_vector_data = np.reshape(input_feature_vector_data, [len(input_feature_vector_data), -1])
				result = sess.run(self.wl_mse[argv_index],
					feed_dict={self.label: train_label,
						self.input_feature_vector_data_holder[argv_index]: input_feature_vector_data,
						self.storage_vector_data_holder[argv_index]: storage_vector_data})
				# mse_result = sess.run(self.mse_result_gpu[argv_index])
				for input_index in range(len(input_feature_vector_data)):
					label = np.argmax(train_label[input_index])
					# shape [ps]
					# mse = mse_result[input_index]
					mse = np.sum(
						np.square(np.subtract(storage_vector_data[label], input_feature_vector_data[input_index])), 1)
					# 0-per_size//2 store max distance points
					# per_size//2-per_size store min distance points
					for distance_index in range(per_size // 2):
						if_put_distance = np.sum(mse[0:per_size // 2], 0) - mse[distance_index]
						current_distance = np.sum(distance[label, 0:per_size // 2, 0:per_size // 2], 1)
						min_index = np.argmin(np.concatenate([[if_put_distance], current_distance], 0))
						if sb_init_map[label] != 0:
							min_index = sb_init_map[label]
						if min_index != 0:
							min_index -= 1
							storage[label, min_index] = np.array(train_input[input_index])
							storage_vector_data[label, min_index] = input_feature_vector_data[input_index]
							mse[min_index] = 0
							distance[label, min_index] = mse
							distance[label, :, min_index] = mse
							break
					for distance_index in range(per_size // 2, per_size):
						if_put_distance = np.sum(mse[per_size // 2:per_size], 0) - mse[distance_index]
						current_distance = np.sum(distance[label, per_size // 2:per_size, per_size // 2:per_size], 1)
						max_index = np.argmax(np.concatenate([[if_put_distance], current_distance], 0))
						if sb_init_map[label] != 0:
							max_index = sb_init_map[label]
						if max_index != 0:
							max_index -= 1
							max_index += per_size // 2
							storage[label, max_index] = np.array(train_input[input_index])
							storage_vector_data[label, max_index] = input_feature_vector_data[input_index]
							mse[max_index] = 0
							distance[label, max_index] = mse
							distance[label, :, max_index] = mse
							break
					if sb_init_map[label] != 0:
						sb_init_map[label] = sb_init_map[label] - 1
				result = sess.run(self.wl_predict[argv_index],
					feed_dict={self.storage_vector_data_holder[argv_index]: storage_vector_data,
						self.input_feature_vector_data_holder[argv_index]: input_feature_vector_data})
			duration = time.time() - start_time
			self.training_count += 1
			print("%d time: %.5f" % (self.training_count, duration))

	def predict(self, sess, predict_input):
		pass

	def save(self, sess, tracer: Tracer, acc: float, loss: float, training_loss: float, training_acc: float,
			appendInfo=""):
		is_best = tracer.is_best(acc, loss)
		tracer.save(sess, acc, loss, training_loss, training_acc, appendInfo)
		if len(self.sb_argv_list) != 0:
			for argv_index in range(len(self.sb_argv_list)):
				file_name = tracer.process_dir + "/sbm_" + str(argv_index) + ".dat"
				np.save(file_name, self.sb_storage_box[argv_index])
				file_name = tracer.process_dir + "/init_map_" + str(argv_index) + ".dat"
				np.save(file_name, self.sb_init_map[argv_index])
				if is_best:
					file_name = tracer.best_dir + "/sbm_" + str(argv_index) + ".dat"
					np.save(file_name, self.sb_storage_box[argv_index])
					file_name = tracer.best_dir + "/init_map_" + str(argv_index) + ".dat"
					np.save(file_name, self.sb_init_map[argv_index])

	def load(self, sess, tracer: Tracer, best=False):
		loaded = tracer.load(sess, best)
		if loaded and len(self.sb_argv_list) != 0:
			for argv_index in range(len(self.sb_argv_list)):
				if best:
					file_name = tracer.best_dir + "/sbm_" + str(argv_index) + ".dat.npy"
					self.sb_storage_box[argv_index] = np.load(file_name)
					file_name = tracer.best_dir + "/init_map_" + str(argv_index) + ".dat.npy"
					self.sb_init_map[argv_index] = np.load(file_name)
				else:
					file_name = tracer.process_dir + "/sbm_" + str(argv_index) + ".dat.npy"
					self.sb_storage_box[argv_index] = np.load(file_name)
					file_name = tracer.process_dir + "/init_map_" + str(argv_index) + ".dat.npy"
					self.sb_init_map[argv_index] = np.load(file_name)
		return loaded

	def batch_norm(self, x, eps=1e-05, decay=0.9, affine=True, name=None):
		def smooth_move(old_value, new_value):
			return tf.assign(old_value, tf.add(decay * old_value, (1 - decay) * new_value))

		with tf.variable_scope(name, default_name='BNScope'):
			x_shape = x.get_shape()
			params_shape = x_shape[-1:]
			axis = list(range(len(x_shape) - 1))
			moving_mean = tf.Variable(tf.constant(0.0, shape=params_shape, dtype=tf.float32), trainable=False,
				name="mean")
			moving_variance = tf.Variable(tf.constant(1.0, shape=params_shape, dtype=tf.float32), trainable=False,
				name="variance")

			def mean_var_with_update():
				mean, variance = tf.nn.moments(x, axis, name='moments')
				with tf.control_dependencies([smooth_move(moving_mean, mean),
					smooth_move(moving_variance, variance)]):
					return tf.identity(moving_mean), tf.identity(moving_variance)

			mean, variance = tf.cond(self.training, mean_var_with_update,
				lambda: (moving_mean, moving_variance))
			if affine:
				beta = tf.Variable(tf.constant(0.0, shape=params_shape, dtype=tf.float32), name="beta")
				gamma = tf.Variable(tf.constant(1.0, shape=params_shape, dtype=tf.float32), name="gamma")
				x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, eps)
			else:
				x = tf.nn.batch_normalization(x, mean, variance, None, None, eps)
			return x

	def allocate_variable(self, shape, stddev=1e-1, trainable=True):
		return tf.Variable(tf.truncated_normal(shape, stddev=stddev, dtype=tf.float32), trainable=trainable)

	def create_conv2d(self, previous_layer, shape, strides, padding, batch_normal=True, stddev=1e-1):
		kernel = tf.nn.conv2d(previous_layer, self.allocate_variable(shape, stddev), strides=strides,
			padding=padding)
		bias = self.allocate_variable(shape=[shape[-1]])
		layer = tf.nn.bias_add(kernel, bias)
		if batch_normal:
			layer = self.batch_norm(layer)
		return tf.nn.relu(layer)

	def create_full_connection(self, previous_layer, weight_shape, stddev=1e-1):
		weight = self.allocate_variable(weight_shape, stddev=stddev)
		bias = self.allocate_variable([weight_shape[-1]], stddev=stddev)
		full_connection = tf.add(tf.matmul(previous_layer, weight), bias)
		return full_connection

	def create_storage_box_module_gpu(self, input_layer, storage_size: int):
		layer_shape = input_layer.get_shape()
		per_size = storage_size * 2
		if len(layer_shape) == 2:
			input_layer = tf.reshape(input_layer, [-1, 1, 1, int(layer_shape[1])])
			layer_shape = input_layer.get_shape()
		layer_shape = [-1, int(layer_shape[1]), int(layer_shape[2]), int(layer_shape[3])]
		self.sb_argv_list.append([input_layer, storage_size])
		self.sb_init_map.append(np.ones([self.label_size], dtype=np.int32) * (storage_size))
		self.sb_storage_box.append(np.zeros(
			[self.label_size, per_size, self.input_shape[1], self.input_shape[2], self.input_shape[3]],
			dtype=np.float32))
		distance_gpu = tf.Variable(tf.zeros([self.label_size, per_size, per_size]), False)
		self.distance_gpu.append(distance_gpu)
		input_feature_vector_data_holder = tf.placeholder(tf.float32, [self.batch_count, None],
			"input_feature_vector_data")
		self.input_feature_vector_data_holder.append(input_feature_vector_data_holder)
		mse = tf.Variable(tf.zeros([self.batch_count, per_size]), False)
		self.mse_result_gpu.append(mse)
		storage_vector_data_holder = tf.placeholder(tf.float32,
			[self.label_size, per_size, None], "storage_vector_data")
		self.storage_vector_data_holder.append(storage_vector_data_holder)
		predict_result_gpu = tf.Variable(tf.zeros([self.batch_count, self.label_size]), False)
		self.predict_result_gpu.append(predict_result_gpu)

		def label_condition(label_index_arg):
			return label_index_arg < tf.constant(self.label_size, dtype=tf.int32)

		def per_storage_condition(per_storage_index_arg):
			return per_storage_index_arg < tf.constant(per_size, dtype=tf.int32)

		def calculate_distance_loop(label_index_arg):
			def iter_per_storage_loop(per_storage_index_arg):
				distance_i_j = tf.reduce_sum(
					tf.square(
						tf.subtract(storage_vector_data_holder[label_index_arg][per_storage_index_arg],
							storage_vector_data_holder[label_index_arg])), 1)
				assign = tf.scatter_nd_update(distance_gpu,
					[[label_index_arg, per_storage_index_arg]], [distance_i_j])
				with tf.control_dependencies([assign]):
					return tf.add(per_storage_index_arg, tf.constant(1, dtype=tf.int32))

			wh_per_storage_loop = tf.while_loop(per_storage_condition, iter_per_storage_loop,
				[tf.constant(0, tf.int32)])
			with tf.control_dependencies([wh_per_storage_loop]):
				return tf.add(label_index_arg, tf.constant(1, tf.int32))

		wl_distance_loop = tf.while_loop(label_condition, calculate_distance_loop,
			[tf.constant(0, dtype=tf.int32)])
		self.wl_distance.append(wl_distance_loop)

		def predict_condition(input_index_arg):
			return input_index_arg < tf.constant(self.batch_count, tf.int32)

		def predict_loop(input_index_arg):
			mse = tf.log(
				tf.reduce_min(
					tf.reduce_sum(
						tf.square(
							tf.subtract(storage_vector_data_holder, input_feature_vector_data_holder[input_index_arg])),
						[2]), [1]) + 1e-6)
			p = tf.exp(-1 * mse) / tf.reduce_sum(tf.exp(-1 * mse), [0])
			update = tf.scatter_nd_update(predict_result_gpu, [[input_index_arg]], [p])
			with tf.control_dependencies([update]):
				return tf.add(input_index_arg, tf.constant(1, tf.int32))

		wl_calculate_predict = tf.while_loop(predict_condition, predict_loop, [tf.constant(0, tf.int32)])
		self.wl_predict.append(wl_calculate_predict)

		def mse_condition(input_index_arg):
			return input_index_arg < tf.constant(self.batch_count, tf.int32)

		def mse_loop(input_index_arg):
			label = tf.cast(tf.argmax(self.label[input_index_arg]), tf.int32)
			# shape [ps]
			per_mse = tf.reduce_sum(
				tf.square(
					tf.subtract(storage_vector_data_holder[label], input_feature_vector_data_holder[input_index_arg])),
				1)
			update = tf.scatter_nd_update(mse, [[input_index_arg]], [per_mse])
			with tf.control_dependencies([update]):
				return tf.add(input_index_arg, tf.constant(1, tf.int32))

		wl_calculate_mse = tf.while_loop(mse_condition, mse_loop, [tf.constant(0, tf.int32)])
		self.wl_mse.append(wl_calculate_mse)

	def add_storage_box_info(self, full_connection):
		if len(self.sb_argv_list) == 0:
			return full_connection
		temp = int(full_connection.get_shape()[-1])
		total_predict = tf.concat(self.predict_result_gpu, 1)
		concat = tf.concat([full_connection, total_predict], 1)
		# concat = tf.concat([full_connection, self.sb_result_holder], 1)
		return self.create_full_connection(concat, [int(concat.get_shape()[-1]), temp])


class ModifiedAlexNet(BaseNet):
	def construct_model(self):
		self.variables.update({'input': self.input, 'label': self.label})
		with tf.name_scope("alexnet"):
			conv1 = self.create_conv2d(self.input, [3, 3, self.input_shape[3], 48], [1, 1, 1, 1], "SAME", False)
			max_pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], "VALID")
			conv2 = self.create_conv2d(max_pool1, [5, 5, 48, 128], [1, 1, 1, 1], "SAME", False)
			max_pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], "VALID")
			conv3 = self.create_conv2d(max_pool2, [3, 3, 128, 192], [1, 1, 1, 1], "SAME", False)
			conv4 = self.create_conv2d(conv3, [3, 3, 192, 192], [1, 1, 1, 1], "SAME", False)
			conv5 = self.create_conv2d(conv4, [3, 3, 192, 192], [1, 1, 1, 1], "SAME", False)
			conv5_shape = conv5.get_shape()
			compression = self.create_conv2d(conv5,
				[int(conv5_shape[1]), int(conv5_shape[2]), int(conv5_shape[3]), 1024], [1, 1, 1, 1], "VALID", False)
			squeeze = tf.squeeze(compression, [1, 2])
			fc1 = self.create_full_connection(squeeze, [1024, 1024])
			fc2 = self.create_full_connection(fc1, [1024, 1024])
			fc3 = self.create_full_connection(fc2, [1024, 10])
			final = fc3
			self.prediction = tf.nn.softmax(final)
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=self.label))
			self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			self.variables.update({'conv5': conv5, 'compression': compression, 'final': final})

	def train(self, sess, train_input, train_label):
		super(ModifiedAlexNet, self).train(sess, train_input, train_label)
		optimizer, = sess.run((self.optimizer,),
			feed_dict={self.input: train_input, self.label: train_label})


class ModifiedAlexNetWithSB(BaseNet):
	def construct_model(self):
		self.decay = tf.placeholder(dtype=tf.float32)
		self.variables.update({'input': self.input, 'label': self.label, 'decay': self.decay})
		with tf.name_scope("alexnetsw"):
			conv1 = self.create_conv2d(self.input, [3, 3, self.input_shape[3], 48], [1, 1, 1, 1], "SAME", False)
			max_pool1 = tf.nn.max_pool(conv1, [1, 3, 3, 1], [1, 2, 2, 1], "VALID")
			conv2 = self.create_conv2d(max_pool1, [5, 5, 48, 128], [1, 1, 1, 1], "SAME", False)
			max_pool2 = tf.nn.max_pool(conv2, [1, 3, 3, 1], [1, 2, 2, 1], "VALID")
			conv3 = self.create_conv2d(max_pool2, [3, 3, 128, 192], [1, 1, 1, 1], "SAME", False)
			self.create_storage_box_module_gpu(conv3, 50)
			conv4 = self.create_conv2d(conv3, [3, 3, 192, 192], [1, 1, 1, 1], "SAME", False)
			conv5 = self.create_conv2d(conv4, [3, 3, 192, 192], [1, 1, 1, 1], "SAME", False)
			self.create_storage_box_module_gpu(conv5, 50)
			conv5_shape = conv5.get_shape()
			compression = self.create_conv2d(conv5,
				[int(conv5_shape[1]), int(conv5_shape[2]), int(conv5_shape[3]), 1024], [1, 1, 1, 1], "VALID", False)
			squeeze = tf.squeeze(compression, [1, 2])
			fc1 = self.create_full_connection(squeeze, [1024, 1024])
			fc2 = self.create_full_connection(fc1, [1024, 10])
			sb = self.add_storage_box_info(fc2)
			final = sb
			self.prediction = tf.nn.softmax(final)
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=self.label))
			# with tf.control_dependencies([ols_update, ols_assign_bias, ols_assign_weight]):
			self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			self.variables.update({'conv5': conv5, 'compression': compression, 'final': final})

	def train(self, sess, train_input, train_label):
		super(ModifiedAlexNetWithSB, self).train(sess, train_input, train_label)
		# , self.sb_result_holder: self.sb_result
		optimizer, = sess.run((self.optimizer,),
			feed_dict={self.input: train_input, self.label: train_label})


class MLP(BaseNet):

	def construct_model(self):
		vector_length = self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
		real_input = tf.reshape(self.input, [-1, vector_length])
		hide1 = self.create_full_connection(real_input, [vector_length, 12])
		hide1 = tf.nn.relu(hide1)
		hide2 = self.create_full_connection(hide1, [12, 8])
		hide2 = tf.nn.relu(hide2)
		hide3 = self.create_full_connection(hide2, [8, 3])
		final = hide3
		self.prediction = tf.nn.softmax(final)
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=self.label))
		self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
		correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def train(self, sess, train_input, train_label):
		super(MLP, self).train(sess, train_input, train_label)
		optimizer, = sess.run((self.optimizer,),
			feed_dict={self.input: train_input, self.label: train_label})


class MLPWithSB(BaseNet):

	def construct_model(self):
		with tf.name_scope("alexnet"):
			vector_length = self.input_shape[1] * self.input_shape[2] * self.input_shape[3]
			real_input = tf.reshape(self.input, [-1, vector_length])
			hide1 = self.create_full_connection(real_input, [vector_length, 12])
			hide1 = tf.nn.relu(hide1)
			self.create_storage_box_module_gpu(hide1, 25)
			hide2 = self.create_full_connection(hide1, [12, 8])
			hide2 = tf.nn.relu(hide2)
			self.create_storage_box_module_gpu(hide2, 25)
			hide3 = self.create_full_connection(hide2, [8, 3])
			sb = self.add_storage_box_info(hide3)
			final = sb
			self.prediction = tf.nn.softmax(final)
			self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final, labels=self.label))
			self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
			correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.label, 1))
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	def train(self, sess, train_input, train_label):
		super(MLPWithSB, self).train(sess, train_input, train_label)
		optimizer, = sess.run((self.optimizer,),
			feed_dict={self.input: train_input, self.label: train_label})


def get_random_block(data, labels, batch_count):
	if len(data) - batch_count < 0:
		batch_count = len(data)
	start_index = np.random.randint(0, len(data) - batch_count)
	# start_index = 0
	data_block = data[start_index:(start_index + batch_count)]
	return np.array(data_block, dtype=np.float32), np.array(labels[start_index:(start_index + batch_count)],
		dtype=np.float32)


def train_model():
	save_root_dir = "F:/store/"
	log_root_dir = "res/log/"
	Alex = "alexnet"
	AlexSB = 'alexswnet'
	Mlp = 'mlp'
	MlpSB = 'mlpsb'
	debug_model = False
	# change model
	model_name = Mlp
	# change model
	print(model_name)
	print("Constructing model...")
	epoch_size = 1000
	if model_name == Alex:
		batch_count = 128
		save_count = 100
		model = ModifiedAlexNet([None, 32, 32, 3], 10, batch_count, 1e-3)
	elif model_name == AlexSB:
		batch_count = 128
		save_count = 100
		model = ModifiedAlexNetWithSB([None, 32, 32, 3], 10, batch_count, 1e-3)
	elif model_name == Mlp:
		batch_count = 20
		save_count = 10
		model = MLP([None, 1, 1, 5], 3, batch_count, 1e-3)
	elif model_name == MlpSB:
		batch_count = 20
		save_count = 10
		model = MLPWithSB([None, 1, 1, 4], 3, batch_count, 1e-3)
	else:
		raise RuntimeError("Unknown model")
	model.construct_model()
	print("Reading data...")
	if model_name == Alex or model_name == AlexSB:
		train_data, train_label, test_data, test_label = Utils.read_cifar10_data()
	elif model_name == Mlp or model_name == MlpSB:
		train_data, train_label, test_data, test_label = Utils.read_iris_data()
	# train_data, train_label, test_data, test_label = Utils.read_cifar10_data()
	else:
		raise RuntimeError("Unknown data set")
	tracer = Tracer(log_root_dir, save_root_dir, model_name, save_count)
	batch_size = (len(train_data) // batch_count) + 1
	with tf.Session() as sess:
		loaded = model.load(sess, tracer)
		if not loaded:
			sess.run(tf.global_variables_initializer())
		if debug_model:
			test_data_block, test_label_block = get_random_block(test_data, test_label, batch_count)
			debug = sess.run((
				model.variables['conv1_mean'], model.variables['conv1_variance'], model.variables['conv2_mean'],
				model.variables['conv2_variance'], model.variables['conv5_mean'],
				model.variables['conv5_variance']),
				feed_dict={model.input: test_data_block, model.label: test_label_block})

		def cal_loss():
			test_data_block, test_label_block = get_random_block(test_data, test_label, batch_count)
			train_data_block, train_label_block = get_random_block(train_data, train_label, batch_count)
			return model.calculate_loss(sess, train_data_block, train_label_block, test_data_block,
				test_label_block)

		accuracy, loss, train_loss, train_acc, predict = cal_loss()
		info = "Before training: accuracy: %.5f loss:%.5f t-loss:%.5f" % (accuracy, loss, train_loss)
		print(info)
		print("Starting train...")
		tracer.start_time_trace()
		for e in range(epoch_size):
			for b in range(batch_size):
				train_data_block, train_label_block = get_random_block(train_data, train_label, batch_count)
				model.train(sess, train_data_block, train_label_block)
				need_save = tracer.inc_count()
				if need_save:
					accuracy, loss, train_loss, train_acc, predict = cal_loss()
					info = "Saved: Epoch %d batch%d " % (e, b)
					model.save(sess, tracer, accuracy, loss, train_loss, train_acc, info)
		accuracy, loss, train_loss, train_acc, predict = cal_loss()
		model.save(sess, tracer, accuracy, loss, train_loss, train_acc)


if __name__ == "__main__":
	train_model()
	print("end")
