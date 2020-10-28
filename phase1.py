import pandas as pd
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
import scipy.stats as stats

gt_2011 = pd.read_csv('gt_2011.csv', header=[0])
gt_2012 = pd.read_csv('gt_2012.csv', header=[0])
gt_2013 = pd.read_csv('gt_2013.csv', header=[0])
gt_2014 = pd.read_csv('gt_2014.csv', header=[0])
gt_2015 = pd.read_csv('gt_2015.csv', header=[0])

x_11 = gt_2011.drop(columns=['NOX', 'CO'])
x_12 = gt_2012.drop(columns=['NOX', 'CO'])
x_13 = gt_2013.drop(columns=['NOX', 'CO'])
x_14 = gt_2014.drop(columns=['NOX', 'CO'])
x_15 = gt_2015.drop(columns=['NOX', 'CO'])
y_11 = gt_2011['NOX']
y_12 = gt_2012['NOX']
y_13 = gt_2013['NOX']
y_14 = gt_2014['NOX']
y_15 = gt_2015['NOX']

linear_regression = LinearRegression(normalize=True)
linear_regression.fit(pd.concat([x_11, x_12], axis=0), pd.concat([y_11, y_12], axis=0))
prediction = linear_regression.predict(x_13)

print('=== Phase 1 : a ===')
print('R^2:', metrics.r2_score(y_13, prediction))
print('Mean absolute error:', metrics.mean_absolute_error(y_13, prediction))
print(stats.spearmanr(y_13, prediction))

linear_regression = LinearRegression(normalize=True)
linear_regression.fit(pd.concat([x_11, x_12, x_13], axis=0), pd.concat([y_11, y_12, y_13], axis=0))
prediction = linear_regression.predict(pd.concat([x_14, x_15]))

print('=== Phase 1 : b ===')
print('R^2:', metrics.r2_score(pd.concat([y_14, y_15]), prediction))
print('Mean absolute error:', metrics.mean_absolute_error(pd.concat([y_14, y_15]), prediction))
print(stats.spearmanr(pd.concat([y_14, y_15]), prediction))
