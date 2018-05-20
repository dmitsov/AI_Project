import tensorflow as tf
import tempfile
import pandas as pd


class SimpleLinearRegression:
	shuffle_size = 10000


	def construct_input_fn(features, labels, batch_size):
		return tf.estimator.inputs.pandas_input_fn(features, labels, batch_size = batch_size, shuffle = True, num_epochs = 10, target_column = 'points')
		
	def construct_standard_regressor(data, result, features, batch_size):
		
		model_dir = tempfile.mkdtemp()
		linearRegressor = tf.estimator.LinearRegressor(model_dir = model_dir, feature_columns = features)
		
		linearRegressor.train(input_fn = SimpleLinearRegression.construct_input_fn(data, result, batch_size))
		
		return linearRegressor
		