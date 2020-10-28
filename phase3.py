# Part 3 Phase 3

import pandas as pd
import numpy as np
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
import scipy.stats as stats
import statistics

gt_2011 = pd.read_csv('gt_2011.csv', header=[0]).drop(columns='CO')
gt_2012 = pd.read_csv('gt_2012.csv', header=[0]).drop(columns='CO')
gt_2013 = pd.read_csv('gt_2013.csv', header=[0]).drop(columns='CO')
gt_2014 = pd.read_csv('gt_2014.csv', header=[0]).drop(columns='CO')
gt_2015 = pd.read_csv('gt_2015.csv', header=[0]).drop(columns='CO')

# Training Set
x_11 = gt_2011.drop(columns=['NOX'])
x_12 = gt_2012.drop(columns=['NOX'])
y_11 = gt_2011['NOX']
y_12 = gt_2012['NOX']

y_13 = gt_2013['NOX']

gt_2013_x = (np.array_split(gt_2013.drop(columns='NOX'), 10))
gt_2013_y = (np.array_split(gt_2013['NOX'], 10))
gt_2014_x = (np.array_split(gt_2014.drop(columns='NOX'), 10))
gt_2014_y = (np.array_split(gt_2014['NOX'], 10))
gt_2015_x = (np.array_split(gt_2015.drop(columns='NOX'), 10))
gt_2015_y = (np.array_split(gt_2015['NOX'], 10))

lr_baseline = LinearRegression(normalize=True)
lr_engineered = LinearRegression(normalize=True)

# Engineered Linear Regression
ATAH_11 = (gt_2011['AT'] * gt_2011['AH']).to_frame()
AT_11 = gt_2011['AT'].pow(2)
AP_11 = gt_2011['AP']
ATAH_12 = (gt_2012['AT'] * gt_2012['AH']).to_frame()
AT_12 = gt_2012['AT'].pow(2)
AP_12 = gt_2012['AP']

# Features:
# AT * AH + AT + AP
f_11 = ATAH_11.assign(AT=AT_11).assign(AP=AP_11)
f_12 = ATAH_12.assign(AT=AT_12).assign(AP=AP_12)

# Initial X and Y values
x_engineered = pd.concat([f_11, f_12])
x_baseline = pd.concat([x_11, x_12], axis=0)
y = pd.concat([y_11, y_12], axis=0)

baseline_predictions = []
engineered_predictions = []


# Results for the validation blocks
for i in range(10):
    # Baseline Linear Regression
    lr_baseline.fit(x_baseline, y)
    prediction = lr_baseline.predict(gt_2013_x[i])

    # Engineered Linear Regression
    lr_engineered.fit(x_engineered, y)
    prediction_engineered = lr_engineered.predict(
        (gt_2013_x[i]['AT'] * gt_2013_x[i]['AH']).to_frame()
            .assign(AT=gt_2013_x[i]['AT'].pow(2))
            .assign(AP=gt_2013_x[i]['AP']))

    print('===== Baseline Results', i+1, ' ====')
    print('R^2:', metrics.r2_score(gt_2013_y[i], prediction))
    print('Mean absolute error:', metrics.mean_absolute_error(gt_2013_y[i], prediction))
    print(stats.spearmanr(gt_2013_y[i], prediction))
    baseline_predictions.extend(prediction)
    print('===== Engineered Results', i+1, '=====')
    print('R^2:', metrics.r2_score(gt_2013_y[i], prediction_engineered))
    print('Mean absolute error:', metrics.mean_absolute_error(gt_2013_y[i], prediction_engineered))
    print(stats.spearmanr(gt_2013_y[i], prediction_engineered))
    engineered_predictions.extend(prediction_engineered)

    # Add latest block
    x_engineered = pd.concat([x_engineered, (gt_2013_x[i]['AT'] * gt_2013_x[i]['AH']).to_frame()
                         .assign(AT=gt_2013_x[i]['AT'].pow(2))
                         .assign(AP=gt_2013_x[i]['AP'])])
    x_baseline = pd.concat([x_baseline, gt_2013_x[i]])
    y = pd.concat([y, gt_2013_y[i]])

# ANOVA test
stat, p = stats.f_oneway(baseline_predictions, y_13)
print('===============ANOVA TEST - BASELINE =====================')
print('F:',stat,' P:', p)

stat, p = stats.f_oneway(engineered_predictions, y_13)
print('===============ANOVA TEST - ENGINEERED ===================')
print('F:',stat,' P:', p)
print('==========================================================')

print('=============== PART 3 - Q5 ==============================')
print('Baseline predictions vs actual values:')
print('R^2:', metrics.r2_score(baseline_predictions, y_13))
print('Mean absolute error:', metrics.mean_absolute_error(baseline_predictions, y_13))
print('Spearmanrank:', stats.spearmanr(baseline_predictions, y_13).correlation)

print('Engineered predictions vs actual values:')
print('R^2:', metrics.r2_score(engineered_predictions, y_13))
print('Mean absolute error:', metrics.mean_absolute_error(engineered_predictions, y_13))
print('Spearmanrank:', stats.spearmanr(engineered_predictions, y_13).correlation)
print('==========================================================')


# Results for the test blocks
# 2014

for i in range(10):
    # Baseline Linear Regression
    lr_baseline.fit(x_baseline, y)
    prediction = lr_baseline.predict(gt_2014_x[i])

    # Engineered Linear Regression
    lr_engineered.fit(x_engineered, y)
    prediction_engineered = lr_engineered.predict(
        (gt_2014_x[i]['AT'] * gt_2014_x[i]['AH']).to_frame()
            .assign(AT=gt_2014_x[i]['AT'].pow(2))
            .assign(AP=gt_2014_x[i]['AP']))

    print('===== Baseline Results', i+1, ' ====')
    print('R^2:', metrics.r2_score(gt_2014_y[i], prediction))
    print('Mean absolute error:', metrics.mean_absolute_error(gt_2014_y[i], prediction))
    print(stats.spearmanr(gt_2014_y[i], prediction))
    baseline_predictions.append(stats.spearmanr(gt_2014_y[i], prediction).correlation)
    print('===== Engineered Results', i+1, '=====')
    print('R^2:', metrics.r2_score(gt_2014_y[i], prediction_engineered))
    print('Mean absolute error:', metrics.mean_absolute_error(gt_2014_y[i], prediction_engineered))
    print(stats.spearmanr(gt_2014_y[i], prediction_engineered))
    engineered_predictions.append(stats.spearmanr(gt_2014_y[i], prediction_engineered).correlation)

    # Add latest block
    x_engineered = pd.concat([x_engineered, (gt_2014_x[i]['AT'] * gt_2014_x[i]['AH']).to_frame()
                         .assign(AT=gt_2014_x[i]['AT'].pow(2))
                         .assign(AP=gt_2014_x[i]['AP'])])
    x_baseline = pd.concat([x_baseline, gt_2014_x[i]])
    y = pd.concat([y, gt_2014_y[i]])

# 2015
for i in range(10):
    # Baseline Linear Regression
    lr_baseline.fit(x_baseline, y)
    prediction = lr_baseline.predict(gt_2015_x[i])

    # Engineered Linear Regression
    lr_engineered.fit(x_engineered, y)
    prediction_engineered = lr_engineered.predict(
        (gt_2015_x[i]['AT'] * gt_2015_x[i]['AH']).to_frame()
            .assign(AT=gt_2015_x[i]['AT'].pow(2))
            .assign(AP=gt_2015_x[i]['AP']))

    print('===== Baseline Results', i+11, ' ====')
    print('R^2:', metrics.r2_score(gt_2015_y[i], prediction))
    print('Mean absolute error:', metrics.mean_absolute_error(gt_2015_y[i], prediction))
    print(stats.spearmanr(gt_2015_y[i], prediction))
    baseline_predictions.append(stats.spearmanr(gt_2015_y[i], prediction).correlation)
    print('===== Engineered Results', i+11, '=====')
    print('R^2:', metrics.r2_score(gt_2015_y[i], prediction_engineered))
    print('Mean absolute error:', metrics.mean_absolute_error(gt_2015_y[i], prediction_engineered))
    print(stats.spearmanr(gt_2015_y[i], prediction_engineered))
    engineered_predictions.append(stats.spearmanr(gt_2015_y[i], prediction_engineered).correlation)

    # Add latest block
    x_engineered = pd.concat([x_engineered, (gt_2015_x[i]['AT'] * gt_2015_x[i]['AH']).to_frame()
                         .assign(AT=gt_2015_x[i]['AT'].pow(2))
                         .assign(AP=gt_2015_x[i]['AP'])])
    x_baseline = pd.concat([x_baseline, gt_2015_x[i]])
    y = pd.concat([y, gt_2015_y[i]])