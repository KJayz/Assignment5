#Part 3 phase 2

import pandas as pd
from sklearn.linear_model import LinearRegression
import scipy.stats as stats

gt_2011 = pd.read_csv('gt_2011.csv', header=[0])
gt_2012 = pd.read_csv('gt_2012.csv', header=[0])
gt_2013 = pd.read_csv('gt_2013.csv', header=[0])
gt_2014 = pd.read_csv('gt_2014.csv', header=[0])
gt_2015 = pd.read_csv('gt_2015.csv', header=[0])

# Headers: AT, AP, AH, AFDP, GTEP, TIT, TAT, TEY, CDP
NOX_test = pd.concat([gt_2014['NOX'], gt_2015['NOX']])

ATAH = (gt_2013['AT'] * gt_2013['AH']).to_frame()
AT = gt_2013['AT'].pow(2)
AP = gt_2013['AP']
# Features:
# AT * AH + AT + AP
f = ATAH.assign(AT=AT).assign(AP=AP)

# Linear Regression
linear_regression = LinearRegression(normalize=True)

highestSpearmanr =0;
for i in range(5):
    f_sample = f.sample(n=2000).sort_index()
    NOX_13 = gt_2013['NOX'].sample(n=2000).sort_index()
    NOX_14 = NOX_test.sample(n=2000).sort_index()
    linear_regression.fit(f_sample, NOX_13)
    prediction = linear_regression.predict(f_sample)
    if(stats.spearmanr(NOX_14, prediction).correlation > highestSpearmanr):
        highestSpearmanr = stats.spearmanr(NOX_14, prediction).correlation
print(highestSpearmanr)
