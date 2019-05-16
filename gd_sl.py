from __future__ import print_function

import numpy as np
import pandas as pd
import math
from matplotlib import cm
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LinearRegression
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format


x = np.random.rand(1000,) * 2 * math.pi
sin = lambda x : x**2 + x
add_normal_noise = lambda x: x + np.random.normal(scale=0.2, size=x.shape)

y = sin(x)
y = add_normal_noise(y)
dataframe = pd.DataFrame({'x':x, 'y':y})
# randomizing the data
dataframe = dataframe.reindex(
	np.random.permutation(dataframe.index))

print(x.shape, y.shape)
lin_reg = LinearRegression()
lin_reg.fit(np.array([x]).T, y)
print(lin_reg.intercept_, lin_reg.coef_)

new_x = np.linspace(0, x.max(), 10000)
new_x = new_x.T
print(new_x)
new_y = lin_reg.predict(np.array([new_x]).T)
print(new_y)
plt.scatter(new_x, new_y, c='r',s=1)
plt.scatter(x, y, c='b', s=1)
plt.show()