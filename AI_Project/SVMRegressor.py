import tensorflow as tf
import pandas as pd
import math
from tensorflow.python.framework import ops
from sklearn.utils import shuffle

class SVMRegressor:

	def _init_(self):
		self.batch_size = 0
		self.A = None
		self.B = None
		self.model = None
		self.loss = None
		self.epsilon = None
		self.loss = None
		self.optimizer = None
		self.session = None
		self.placeholder_data = None
		self.placeholder_value = None

	def ConstructSVM(self, column_number, value, batch_size, constant_epsilon = 0.5, learning_rate = 0.075):
		ops.reset_default_graph()
		
		self.session = tf.Session()

		self.batch_size = batch_size
		self.placeholder_data = tf.placeholder(dtype = tf.float32, shape = [None, column_number])
		self.placeholder_value = tf.placeholder(dtype = tf.float32, shape = [None, 1]);
		
		self.A = tf.Variable(tf.random_normal(shape = [len(data.columns_), 1]), name = 'A')
		self.B = tf.Variable(tf. random_normal(shape = [1, 1]), name = 'B')
		
		self.epsilon = tf.constant(constant_epsilon)

		self.model = tf.add(tf.matmul(placeholder_data, self.A), self.B)

		self.loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.substract(self.model, self.placeholder_value)), self.epsilon)))
		
		self.optimizer = tf.train.GradientDescentOptimizer(learning_rate)
	
	def TrainSVM(data, values_column_name, test_data, num_epochs):
		init = tf.global_variables_initializer()
		iterations = math.ceil(len(data) / self.batch_size)
		
		with tf.name_scope("summaries"):
			mean_A = tf.reduce_mean(self.A)
			tf.summary.scalar('A', self.A)
			tf.summary.scalar('mean', mean)
			stddev_A = tf.sqrt(tf.reduce_mean(tf.square(self.A - mean_A)))
			tf.summary.scalar('stddev', stddev_A)
			tf.summary.scalar('max', tf.reduce_max(self.A))
			tf.summary.scalar('min', tf.reduce_min(self.A))
			tf.summary.histogram('histogram', self.A)
			
			mean_B = tf.readuce_mean(self.B)
			stddev_B = tf.sqrt(tf.reduce_mean(tf.square(self.B - mean_B)))
			tf.summary.scalar('B', self.B)
			tf.summary.scalar('mean', mean_B)
			tf.summary.scalar('stddev', stddev_B)
			tf.summary.scalar('max', tf.reduce_max(self.B))
			tf.summary.scalar('min', tf.reduce_min(self.B))
			tf.summary.histogram('histogram', self.B)
			
		tf.summary.scalar(self.loss, tf.float32)

		merged = tf.summary.merge_all()
		self.train_writer = tf.summary.FileWriter(".\train", self.session.graph)
		self.test_writer = tf.summary.FileWriter(".\test", self.session.graph)
		
		self.session.run(init)
		test_values = test_data[values_column_name]
		test_data = test_data.drop(values_column_name)

		for i in range(num_epochs):
			data = shuffle(data)
			for j in range(iterations):
				batch_data = None
				batch_values = None
				if(j == iterations - 1):
					batch_data = data.iloc[(j - 1) * self.batch_size,  len(data)]
					
				else:
					batch_data = data.iloc[j * self.batch_size : (j + 1) * batch_size]
				
				batch_values = batch_data[values_column_name]
				batch_data = batch_data.drop(values_column_name)
				
				self.session.run(self.optimizer, feed_dict = {placeholder_data: batch_data, placeholder_value: batch_values})

				if j % 50 == 0:
					summary_test, test_loss = self.session.run([merged, self.loss], feed_dict = {placeholder_data: test_data, placeholder_value: test_values})
					self.test_writer.add_summary(summary_test, j)
					print("Test loss: %s" % str(test_loss))
				
				summary_train, _ = self.session.run([merged, self.loss], feed_dict = {placeholder_data: batch_data, placeholder_value: batch_values})
				self.train_writer.add_summary(summary_train, j)
		return self.model