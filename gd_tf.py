from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import math
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


x = np.random.rand(1000,) * 2 * math.pi
# sin = lambda x: np.sin(x)
sin = lambda x : x**2 + x
add_normal_noise = lambda x: x + np.random.normal(scale=0.2, size=x.shape)

y = sin(x)
y = add_normal_noise(y)
dataframe = pd.DataFrame({'x':x, 'y':y})
# randomizing the data
dataframe = dataframe.reindex(
	np.random.permutation(dataframe.index))

# define feature column
feature = dataframe[['x']]
feature_columns = [tf.feature_column.numeric_column('x')]
# define target
targets = dataframe['y']
# Create a Linear Regressor
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

# configuring the linear regressor
linear_regressor = tf.estimator.LinearRegressor(
	feature_columns=feature_columns,
	optimizer=optimizer
)

def input_fucntion(features, targets, batch_size=1, shuffle=True, num_epochs=None):
	features = {key:np.array(value) for key,value in dict(features).items()}
	ds = Dataset.from_tensor_slices((features, targets))
	ds = ds.batch(batch_size).repeat(num_epochs)
	if shuffle:
		ds = ds.shuffle(buffer_size=10000)
	features, labels = ds.make_one_shot_iterator().get_next()
	return features, labels

# Train the model
linear_regressor.train(
	input_fn=lambda: input_fucntion(feature, targets),
	steps=100
	)
predict_function = lambda: input_fucntion(feature, targets, num_epochs=1, shuffle=False)
predictions = linear_regressor.predict(input_fn=predict_function)
predictions = np.array([item['predictions'][0] for item in predictions])
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error)


# output = pd.DataFrame()
# output['predictions'] = pd.Series(predictions)
# output['targets'] = pd.Series(targets)

sample = dataframe.sample(n=200)
x_0 = sample['x'].min()
x_1 = sample['x'].max()

weights = linear_regressor.get_variable_value('linear/linear_model/x/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')

y_0 = weights * x_0 + bias
y_1 = weights * x_1 + bias

plt.plot([x_0, x_1], [y_0, y_1], c='r')
plt.scatter(sample['x'], sample['y'])
plt.show()


def train(learning_rate, steps, batch_size, input_feature='x'):
	periods = 10
	steps_per_period = steps/periods
	feature = input_feature
	feature_data = dataframe[[feature]]
	label = 'y'
	targets = dataframe[label]
	
	feature_columns = [tf.feature_column.numeric_column(feature)]
	
	training_input_function = lambda: input_fucntion(feature_data, targets, batch_size=batch_size)
	prediction_input_function = lambda: input_function(feature_data, targets, num_epochs=1, shuffle=False)
	
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
	optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
	linear_regressor = tf.estimator.LinearRegressor(
	  feature_columns=feature_columns,
	  optimizer=optimizer
	)
	for period in range(periods):
		linear_regressor.train(
			input_fn=training_input_function,
			steps=steps_per_period
		)
		x_extents = np.array([0, sample[feature].max()])
		weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
		bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
		# x_extents = (y_extents - bias)/weight
		y_extents = x_extents * weight + bias
		# x_extents = np.maximum(np.maximum(x_extents,
		# 	sample[feature].max()),
		# 	sample[feature].min()
		# )
		# y_extents = weight * x_extents + bias
		if period == periods - 1:
			plt.scatter(np.array(sample[feature]), np.array(sample[label]), c='g')
			plt.plot(x_extents, y_extents)
	print("model training finished")
	
train(
	learning_rate = 0.1,
	steps = 100,
	batch_size = 1
)
print("program terminating")
plt.show()
	
		
	
