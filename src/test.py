from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from matplotlib import pyplot
import numpy as np
import pandas as pd
import talib
import sys
import six
import pywt
import matplotlib.pyplot as plt


# parsing date
def parser(x):
  return datetime.strptime(x,'%Y-%m-%d')


#########################
# read data source files
#########################
prices = read_csv('data/HSI.csv',header=0,parse_dates=[0],date_parser=parser,index_col=0)
prices.dropna(inplace=True)
#print(prices.isnull().values.any())
prices.columns = ['open','high','low','close','adjclose','volume']
prices.index.name = 'date'
#print(dataset.head())


#------------------------ construct features examples ------------------------#

############################
# create features dataframe
############################
features = DataFrame(index=prices.index)
features['volume_change_ratio'] = prices.volume.diff(1) / prices.shift(1).volume
features['momentum_5_day'] = prices.close.pct_change(5) 
features['intraday_chg'] = (prices.close.shift(0) - prices.open.shift(0))/prices.open.shift(0)
features['day_of_week'] = features.index.get_level_values('date').weekday
features['day_of_month'] = features.index.get_level_values('date').day
features.dropna(inplace=True)
#print(features.isnull().values.any())
#print(features.tail(10))


###########################
# create outcome dataframe
###########################
outcomes = DataFrame(index=prices.index)
# next day's opening change
outcomes['open_1'] = prices.open.shift(-1)/prices.close.shift(0)-1
# next day's closing change
outcomes['close_1'] = prices.close.pct_change(-1)
outcomes['close_5'] = prices.close.pct_change(-5) 
outcomes.dropna(inplace=True)
#print(outcomes.isnull().values.any())
#print(outcomes.tail(15))


#########################################################################################
# sample codes: construct features from the data and store into multiple feature columns
#########################################################################################
features = pd.DataFrame(index=prices.index).sort_index()
features['f01'] = prices.close/prices.open-1 # daily return
features['f02'] = prices.open/prices.groupby(level='symbol').close.shift(1)-1 
features.tail()

##########################################
# sample codes: put values into log space
##########################################
features['f03'] = prices.volume.apply(np.log) # log of daily volume

##########################################
# sample codes: how a value is changing
##########################################
features['f03'] = prices.groupby(level='symbol').volume.diff() # change since prior day
features['f04'] = prices.groupby(level='symbol').volume.diff(50) # change since 50 days prior

###################################
# sample codes: get rate of change
###################################
pct_chg_fxn = lambda x: x.pct_change()
features['f05'] = prices.groupby(level='symbol').volume.apply(pct_chg_fxn) 

###################################
# sample codes: get moving average
###################################
# log of 5 day moving average of volume
ma_5 = lambda x: x.rolling(5).mean()
features['f06'] = prices.volume.groupby(level='symbol').apply(ma_5)\
.apply(np.log) 

# daily volume vs. 200 day moving average
ma_200 = lambda x: x.rolling(200).mean()
features['f07'] = prices.volume/ prices.volume.groupby(level='symbol')\
.apply(ma_200)-1

# daily closing price vs. 50 day exponential moving avg
ema_50 = lambda x: x.ewm(span=50).mean()
features['f08'] = prices.close/ prices.close.groupby(level='symbol')\
.apply(ema_50)-1

#############################
# sample codes: get z-score
#############################
zscore_fxn = lambda x: (x - x.mean()) / x.std()
features['f09'] =prices.groupby(level='symbol').close.apply(zscore_fxn)
features.f09.unstack().plot.kde(title='Z-Scores (not quite accurate)')

zscore_fun_improved = lambda x: (x - x.rolling(window=200, min_periods=20).mean())\
/ x.rolling(window=200, min_periods=20).std()
features['f10'] =prices.groupby(level='symbol').close.apply(zscore_fun_improved)
features.f10.unstack().plot.kde(title='Z-Scores (accurate)')

###############################
# sample codes: get percentile
###############################
rollrank_fxn = lambda x: x.rolling(200,min_periods=20)\
.apply(lambda x: pd.Series(x).rank(pct=True)[0])
features['f11'] = prices.groupby(level='symbol').volume.apply(rollrank_fxn)

# ranking
features['f12'] = features['f07'].dropna().groupby(level='date').rank(pct=True) 

####################################################
# sample codes: technical analysis using TA library
####################################################
import ta # technical analysis library: https://technical-analysis-library-in-python.readthedocs.io/en/latest/
# money flow index (14 day)
features['f13'] = ta.momentum.money_flow_index(prices.high, 
                                               prices.low, prices.close, \
                                               prices.volume, n=14, fillna=False)
# mean-centered money flow index
features['f14'] = features['f13'] - features['f13']\
.rolling(200,min_periods=20).mean()

#########################
# sample codes: binning
#########################
n_bins = 10
bin_fxn = lambda y: pd.qcut(y,q=n_bins,labels = range(1,n_bins+1))
features['f15'] = prices.volume.groupby(level='symbol').apply(bin_fxn)

#########################
# sample codes: signing
#########################
features['f16'] = features['f05'].apply(np.sign)

############################
# sample codes: plus-minus
############################
plus_minus_fxn = lambda x: x.rolling(20).sum()
features['f17'] = features['f16'].groupby(level='symbol').apply(plus_minus_fxn)

#################################
# sample codes: one hot encoding
#################################
month_of_year = prices.index.get_level_values(level='date').month
one_hot_frame = pd.DataFrame(pd.get_dummies(month_of_year))
one_hot_frame.index = prices.index # Careful!  This is forcing index values without usual pandas alignments!

# create column names 
begin_num = int(features.columns[-1][-2:]) + 1 #first available feature
feat_names = ['f'+str(num) for num in list(range(begin_num,begin_num+12,1))]

# rename columns and merge
one_hot_frame.columns = feat_names
features = features.join(one_hot_frame)
features.iloc[:,-12:].tail()


#------------------------ training examples ------------------------#

#################################
# train with a linear regression
#################################
from sklearn.linear_model import LinearRegression

# combine features dataframe and outcomes dataframe
# first, create y (a series) and X (a dataframe), with only rows where 
# a valid value exists for both y and X
y = outcomes.close_1
X = features
Xy = X.join(y).dropna()
y = Xy[y.name]
X = Xy[X.columns]
print(y.shape)
print(X.shape)

model = LinearRegression()
model.fit(X,y)
print("Model RSQ: "+ str(model.score(X,y)))

print("Coefficients: ")
pd.Series(model.coef_,index=X.columns).sort_values(ascending=False)


#############################
# train with a random forest
#############################
# combine features dataframe and outcomes dataframe
# first, create y (a series) and X (a dataframe), with only rows where 
# a valid value exists for both y and X
y = outcomes.open_1
X = features
Xy = X.join(y).dropna()
y = Xy[y.name]
X = Xy[X.columns]
print(y.shape)
print(X.shape)

model = RandomForestRegressor(max_features=3)
model.fit(X,y)
print("Model Score: "+ str(model.score(X,y)))

print("Feature Importance: ")
pd.Series(model.feature_importances_,index=X.columns).sort_values(ascending=False)

#########################
# output to CSV and exit
########################
#dataset = dataset.dropna()
#dataset.to_csv('data/preprocessed_features.csv')
sys.exit()
