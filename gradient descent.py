from __future__ import print_function

import pandas
from tensorflow.python.data import Dataset
from IPython import display
import math
import numpy
import tensorflow as tf

from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
tf.version

pandas.options.display.float_format = '{:.1f}'.format

dataframe = pandas.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")

feature = dataframe[['total_rooms']]
feature_columns = [tf.feature_column.numeric_column("total_rooms")]
targets = dataframe['median_house_value']

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000001)
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

# creating the model
linear_regressor = tf.estimator.LinearRegressor(
    feature_columns=feature_columns,
    optimizer=optimizer
)

# defineing the input function for the input in the model


def input_function(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    features = {key: numpy.array(value)
                for key, value in dict(features).items()}
    ds = Dataset.from_tensor_slices((features, targets))
    ds = ds.batch(batch_size).repeat(num_epochs)
    if shuffle:
        ds = ds.shuffle(buffer_size=10000)
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# training the model
linear_regressor.train(
    input_fn=lambda: input_function(feature, targets), steps=100)


def prediction_input_function(): return input_function(
    feature, targets, num_epochs=1, shuffle=False)


predictions = linear_regressor.predict(input_fn=prediction_input_function)
predictions = numpy.array([item['predictions'][0] for item in predictions])
# Print Mean Squared Error and Root Mean Squared Error.
mean_squared_error = metrics.mean_squared_error(predictions, targets)
root_mean_squared_error = math.sqrt(mean_squared_error)
print("Mean Squared Error (on training data): %0.3f" % mean_squared_error)
print("Root Mean Squared Error (on training data): %0.3f" %
      root_mean_squared_error)

# comparing the performance with the dataset
min_house_value = dataframe["median_house_value"].min()
max_house_value = dataframe["median_house_value"].max()
min_max_difference = max_house_value - min_house_value

print("Min. Median House Value: %0.3f" % min_house_value)
print("Max. Median House Value: %0.3f" % max_house_value)
print("Difference between Min. and Max.: %0.3f" % min_max_difference)
print("Root Mean Squared Error: %0.3f" % root_mean_squared_error)

calibration_data = pandas.DataFrame()
calibration_data["predictions"] = pandas.Series(predictions)
calibration_data["targets"] = pandas.Series(targets)
calibration_data.describe()

sample = dataframe.sample(n=300)

x_0 = sample["total_rooms"].min()
x_1 = sample["total_rooms"].max()
# Retrieve the final weight and bias generated during training.
weight = linear_regressor.get_variable_value(
    'linear/linear_model/total_rooms/weights')[0]
bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
# Get the predicted median_house_values for the min and max total_rooms values.
y_0 = weight * x_0 + bias
y_1 = weight * x_1 + bias

# Plot our regression line from (x_0, y_0) to (x_1, y_1).
plt.plot([x_0, x_1], [y_0, y_1], c='r')
# Label the graph axes.
plt.ylabel("median_house_value")
plt.xlabel("total_rooms")
# Plot a scatter plot from our data sample.
plt.scatter(sample["total_rooms"], sample["median_house_value"])
# Display graph.
plt.show()

def train(learning_rate, steps, batch_size, input_feature="total_rooms"):
    periods = 10
    steps_per_periods = steps/periods
    feature = input_feature
    feature_data = dataframe[[feature]]
    label = "median_house_value"
    targets = dataframe[label]
    feature_column = [tf.feature_column.numeric_column(feature)]

    training_function = lambda:input_function(
        feature_data, targets, batch_size=batch_size)
    prediction_input_function = lambda: input_function(
        feature_data, targets, num_epochs=1, shuffle=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=feature_column,
        optimizer=optimizer
        )
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    sample = dataframe.sample(n=200)
    plt.scatter(sample[feature], sample[label])
    colors = [cm.coolwarm(x) for x in numpy.linspace(-1, 1, periods)]
    print("Training model...")
    print("RMSE (on training data):")
    root_mean_squared_errors = []
    for period in range(periods):
        linear_regressor.train(
            input_fn=training_function,
            steps=steps_per_periods
        )
        predictions = linear_regressor.predict(input_fn=prediction_input_function)
        predictions = numpy.array([item['predictions'][0] for item in predictions])
        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(predictions, targets))
        print(" period %02d : %-.2f" % (period, root_mean_squared_error))
        root_mean_squared_errors.append(root_mean_squared_error)
        y_extents = numpy.array([0, sample[label].max()])
        weight = linear_regressor.get_variable_value('linear/linear_model/%s/weights' % input_feature)[0]
        bias = linear_regressor.get_variable_value('linear/linear_model/bias_weights')
        x_extents = (y_extents - bias) / weight
        x_extents = numpy.maximum(
            numpy.minimum(x_extents,
                sample[feature].max()),
            sample[feature].min())
        y_extents = weight * x_extents + bias
        plt.plot(x_extents, y_extents, color=colors[period]) 
        plt.show()
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.subplot(1, 2, 2)
    plt.ylabel('RMSE')
    plt.xlabel('Periods')
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(root_mean_squared_error)
    # Output a table with calibration data.
    calibration_data = pandas.DataFrame()
    calibration_data["predictions"] = pandas.Series(predictions)
    calibration_data["targets"] = pandas.Series(targets)
    display.display(calibration_data.describe())

    print("Final RMSE (on training data): %0.2f" % root_mean_squared_error)

train(
    learning_rate=0.00001,
    steps=100,
    batch_size=1
)