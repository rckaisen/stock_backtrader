############################################
# Preprocessing features and output to file
#
# Output file: preprocessed_features.csv
############################################

####
# open
# close
# low
# high
# volume
# RSI(14)
# MACD(12,26), EMA(9), Divergence
# SMA(10), SMA(20), SMA(50), SMA(100), SMA(200)
# OBV
# ROC
####


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
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer

# parsing date
def parser(x):
  return datetime.strptime(x,'%Y-%m-%d')


# Class labeling for positive and negative number 
def classlabeling(diff):
  if (diff >= 0): 
    return 1
  else: 
    return 0  

# Calculate percentage change
pct_chg_fxn = lambda x: x.pct_change()

########################
# Candlesticks Patterns 
########################
def getCandleSticks(candlesticks):
  #print(candlesticks.shape)

  start = 5
  end = 66
  candles = candlesticks.iloc[:,start:end]
  #print(candles.shape)

  candles = candles.clip(-100,100)
  candles = candles.replace(to_replace=100, value=1)
  candles = candles.replace(to_replace=-100, value=-1)
  candles = candles.sum(axis=1)
  dataset['candlesticks'] = candles.clip(-3,3)
  dataset['candlesticks'] = dataset['candlesticks'].shift(1).fillna(0).astype('int')
  #print(dataset.head(30))


##############
# RSI Change
##############
def categorizeRSI(rsi):
  if (rsi >= 70): 
    return 2
  elif (rsi <= 30):
    return 0
  else: 
    return 1

def RSI_1hot(rsi):
  encoder = OneHotEncoder()
  lb = LabelBinarizer()
  df_cat_1hot = encoder.fit_transform(rsi)
  print("1hot: ", df_cat_1hot.toarray())
  x = lb.fit_transform(np.array(df_cat_1hot.toarray()))
  print("lb: ", x)
  y = lb.inverse_transform(x)
  print("lb: ", y)
  #print("check: ", encoder.transform([[0,1], [1,2], [2,1], [1,0], [0,2], [2,0], [0,0], [1,1], [2,2]]).toarray())
  #return df_cat_1hot

def convert(v, cat):
  if (v == True): 
    return cat
  elif (v == False):
    return 0

def relabel(val):
  if (val == 1):
    #return "noChange_below30"
    return 1
  elif (val == 2):
    #return "noChange_btw30and70"
    return 2
  elif (val == 3):
    #return "noChange_above70"
    return 3
  elif (val == 4):
    #return "below30_to_btw30and70"
    return 4
  elif (val == 5):
    #return "btw30and70_to_above70"
    return 5
  elif (val == 6):
    #return "above70_to_btw30and70"
    return 6
  elif (val == 7):
    #return "btw30and70_to_below30"  
    return 7
  elif (val == 8):
    #return "below30_to_above70"
    return 8
  elif (val == 9):
    #return "above70_to_below30"
    return 9

def RSI_patterns(a, b):
  # prev=0, curr=0 (both previous RSI and current RSI are below 30)
  pattern1 = np.logical_and(a==0, b==0)
  pattern1 = pattern1.apply(lambda v: convert(v, 1))
  # prev=1, curr=1 (both previous RSI and current RSI are between 30 and 70)
  pattern2 = np.logical_and(a==1, b==1)
  pattern2 = pattern2.apply(lambda v: convert(v, 2))
  # prev=2, curr=2 (both previous RSI and current RSI are above 70)
  pattern3 = np.logical_and(a==2, b==2)
  pattern3 = pattern3.apply(lambda v: convert(v, 3))
  # prev=0, curr=1 (previous RSI below 30 goes up to current RSI 30~70 range)
  pattern4 = np.logical_and(a==0, b==1)
  pattern4 = pattern4.apply(lambda v: convert(v, 4))
  # prev=1, curr=2 (previous RSI 30~70 range goes up to current RSI above 70)
  pattern5 = np.logical_and(a==1, b==2)
  pattern5 = pattern5.apply(lambda v: convert(v, 5))
  # prev=2, curr=1 (previous RSI above 70 goes down to current RSI 30~70 range)
  pattern6 = np.logical_and(a==2, b==1)
  pattern6 = pattern6.apply(lambda v: convert(v, 6))
  # prev=1, curr=0 (previous RSI 30~70 range goes down to current RSI below 30)
  pattern7 = np.logical_and(a==1, b==0)
  pattern7 = pattern7.apply(lambda v: convert(v, 7))
  # prev=0, curr=2 (previous RSI below 30 goes up to current RSI above 70)
  pattern8 = np.logical_and(a==0, b==2)
  pattern8 = pattern8.apply(lambda v: convert(v, 8))
  # prev=2, curr=0 (previous RSI above 70 goes down to current RSI below 30)
  pattern9 = np.logical_and(a==2, b==0)
  pattern9 = pattern9.apply(lambda v: convert(v, 9))
  # add all columns element-wise 
  dataset['RSI_5day_vs_prevday'] = np.add(np.add(np.add(np.add(np.add(np.add(np.add(np.add(pattern1, pattern2), pattern3), pattern4), pattern5), pattern6), pattern7), pattern8), pattern9)
  dataset['RSI_5day_vs_prevday'] = dataset['RSI_5day_vs_prevday'].shift(-5).fillna(1).astype('int')
  dataset['RSI_5day_vs_prevday'] = dataset['RSI_5day_vs_prevday'].apply(lambda v: relabel(v))
  #return pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9
  #return output 

def getRSIChange(dataset):
  # Compute RSI of close price
  rsi = talib.RSI(dataset["Close"].values, timeperiod=14)
  #print(rsi.shape)
  
  # Add columns of RSI and RSI Change dataset
  # Use diff() to calculate difference
  # Use pct_change() to calculate pct_change() (not used in current setting)
  # Use categorizeRSI() to classify RSI value level (not used in current setting)
  dataset['RSI'] = rsi
  dataset['RSI_PREV'] = dataset['RSI'].shift(1).fillna(0).astype('int')
  dataset['RSI_PREV_CAT'] = dataset['RSI_PREV'].apply(lambda d: categorizeRSI(d))
  #dataset['RSI_CAT'] = dataset['RSI'].apply(lambda d: categorizeRSI(d))
  #dataset['RSI_CAT_PREV'] = dataset['RSI_CAT'].shift(1).fillna(0).astype('int') 
  #dataset['RSI_CAT_NEXT'] = dataset['RSI_CAT'].shift(-1).fillna(0).astype('int')
  #dataset['RSI_CHG'] = dataset['RSI'].diff()
  #dataset['RSI_CHG_CAT'] = dataset['RSI_CHG'].apply(lambda d: classlabeling(d))
  #dataset['RSI_CHG_ROC'] = dataset['RSI_CHG'].pct_change()
  dataset['RSI_AVG_5DAYS'] = talib.SMA(dataset['RSI_PREV'], timeperiod=5).fillna(0).astype('int').shift(-5) 
  dataset['RSI_AVG_5DAYS_CAT'] = dataset['RSI_AVG_5DAYS'].apply(lambda d: categorizeRSI(d))
  # compare RSI_AVG_5DAYS_CAT and RSI_CAT_PREV
  dataset = dataset.dropna()
  

  # df_encoded, df_categories = dataset[['RSI_CAT_PREV','RSI_CAT']].factorize()
  # print("2: ", df_encoded[:])
  # print("3: ", df_categories[:])
  #RSI_1hot(dataset[['RSI_CAT_PREV','RSI_CAT']])
  #dataset['NoChange_below30'], dataset['NoChange_normal'], dataset['NoChange_above70'], dataset['below30_to_normal'], dataset['normal_to_above70'], dataset['above70_to_normal'], dataset['normal_to_below30'], dataset['below30_to_above70'], dataset['above70_to_below30'] = check(dataset['RSI_AVG_5DAYS_CAT'], dataset['RSI_PREV_CAT'])
  #dataset['RSI_5day_vs_prevday'] = RSI_patterns(dataset['RSI_AVG_5DAYS_CAT'], dataset['RSI_PREV_CAT']) 
  RSI_patterns(dataset['RSI_AVG_5DAYS_CAT'], dataset['RSI_PREV_CAT']) 
  # skip first 5 rows because of RSI average 5 days
  #print(dataset[5:]) 

  #plt.hist(dataset['RSI'], bins = 30)
  #plt.xlabel("Number")
  #plt.ylabel("Frequency")
  #plt.show()

#####################
# Close Price Change
#####################
def getCloseChange(dataset):
  dataset['CLOSE_CHG'] = dataset['Close'].diff()
  dataset['CLOSE_CHG_CAT'] = dataset['CLOSE_CHG'].shift(1).apply(lambda d: classlabeling(d))
  #dataset['Close_LOG'] = np.log(dataset['Close'])
  #dataset = dataset.dropna()
  #print(dataset.head(30))
   

#########################################
# feature: Rate of change of close price
#########################################
def roc_slope(df, days):
  sma = talib.SMA(df, timeperiod=days)
  rocOfValue = talib.ROCP(sma, timeperiod=days)
  rocOfValueChange = talib.ROCP(rocOfValue, timeperiod=days)
  rocOfValueChangeCat = rocOfValueChange.apply(lambda d: 1 if d>=0 else -1)
  return rocOfValueChange, rocOfValueChangeCat	

def getRocOFCloseChange(dataset):
  # Calculate ROC of value change of SMA of 2 days  
  dataset['rocOFCloseChange'], dataset['rocOFCloseChangeCat'] = roc_slope(dataset['Close'], 2)
  #dataset = dataset.dropna()
  #print(dataset.head(30))

################
# feature: SMA
################
def smaDiff(d1, d2, l1, l2):
  diff1 = (d1 - d2).apply(lambda d: 1 if  d>=0 else -1)  # previous day
  dataset['diff1'] = diff1
  diff2 = diff1.shift(1).apply(lambda d: 1 if  d>=0 else -1)  # previous two days
  dataset['diff2'] = diff2

  dataset['test1'] = ((diff1 == 1) & (diff2 == 1)).apply(lambda x: 1 if x==True else 0) # sma5 above sma14
  dataset['test2'] = ((diff1 == 1) & (diff2 == -1)).apply(lambda x: 2 if x==True else 0) # sma5 cross below sma14
  dataset['test3'] = ((diff1 == -1) & (diff2 == 1)).apply(lambda x: 3 if x==True else 0) # sma5 cross above sma14
  dataset['test4'] = ((diff1 == -1) & (diff2 == -1)).apply(lambda x: 4 if x==True else 0) # sma5 below sma14
  #dataset['smaDiff_5_14'] = (np.add(np.add(np.add(dataset['test1'], dataset['test2']), dataset['test3']), dataset['test4'])).apply(lambda x: 'sma5_above_sma14' if x==1 else 'sma_cross_below_sma14' if x==2 else 'sma5_cross_above_sma14' if x==3 else 'sma5_below_sma14')
  dataset['smaDiff_5_14'] = (np.add(np.add(np.add(dataset['test1'], dataset['test2']), dataset['test3']), dataset['test4']))

  #result = (diff1 - diff2).apply(lambda d: l1+"_cross_above_"+l2 if d==-2 else (l1+"_cross_below_"+l2 if d==2 else "sma_normal"))
  #print(dataset.head(50))
  #return result



def getSMA(dataset):
  dataset['sma5'] = talib.SMA(dataset.Close, timeperiod=5)
  dataset['sma14'] = talib.SMA(dataset.Close, timeperiod=14)
  #dataset['sma50'] = talib.SMA(dataset.Close, timeperiod=50)
  #dataset['sma200'] = talib.SMA(dataset.Close, timeperiod=200)
  #dataset.dropna(inplace=True)
  smaDiff(dataset['sma5'], dataset['sma14'], 'sma5', 'sma14')
  #dataset['smaDiff_5_14'] = smaDiff(dataset['sma5'], dataset['sma14'], 'sma5', 'sma14')
  #dataset['smaDiff_5_50'] = smaDiff(dataset['sma5'], dataset['sma50'], 'sma5', 'sma50')
  #dataset['smaDiff_5_200'] = smaDiff(dataset['sma5'], dataset['sma200'], 'sma5', 'sma200')
  #dataset['smaDiff_14_50'] = smaDiff(dataset['sma14'], dataset['sma50'], 'sma14', 'sma50')
  #dataset['smaDiff_14_200'] = smaDiff(dataset['sma14'], dataset['sma200'], 'sma14', 'sma200')
  #dataset['smaDiff_50_200'] = smaDiff(dataset['sma50'], dataset['sma200'], 'sma50', 'sma200')
  #print(dataset.head(30))


##################
# feature: Volume
##################
def getVolumnChange(dataset):
  dataset['OBV'] = talib.OBV(dataset["Close"], dataset["Volume"])
  dataset['Volume_CHG'] = dataset['Volume'].shift(1).pct_change()
  dataset['prev1_volume'] = dataset['Volume'].shift(1)
  dataset['prev2_volume'] = dataset['Volume'].shift(2)
  dataset['prev3_volume'] = dataset['Volume'].shift(3)
  dataset['sma5_volume'] = talib.SMA(dataset.prev1_volume, timeperiod=5)

  check_vol_uptrend(dataset['prev1_volume'], dataset['prev2_volume'], dataset['prev3_volume'], dataset['sma5_volume'])
  check_vol_downtrend(dataset['prev1_volume'], dataset['prev2_volume'], dataset['prev3_volume'], dataset['sma5_volume'])
  dataset.dropna(inplace=True)
  combined_volume_trends(dataset['vol_up_trend'], dataset['vol_down_trend'])

# previous 3 days volume greater than sma5_volume
def check_vol_uptrend(d1, d2, d3, d4):
  dataset['vol_up_trend'] = np.logical_and(np.logical_and(d1>d4, d2>d4), d3>d4)

# previous 3 days volume less than sma5_volume
def check_vol_downtrend(d1, d2, d3, d4):
  dataset['vol_down_trend'] = np.logical_and(np.logical_and(d1<d4, d2<d4), d3<d4)

def combined_volume_trends(d1, d2):
  dataset['vol_test1'] = ((d1 == False) & (d2 == False)).apply(lambda x: 0 if x==True else 0)  
  dataset['vol_test2'] = ((d1 == True) & (d2 == False)).apply(lambda x: 1 if x==True else 0)
  dataset['vol_test3'] = ((d1 == False) & (d2 == True)).apply(lambda x: 2 if x==True else 0) 
  dataset['vol_trend'] = (np.add(np.add(dataset['vol_test1'], dataset['vol_test2']), dataset['vol_test3'])).apply(lambda x: 'vol_normal' if x==0 else 'vol_up_trend' if x==1 else 'vol_down_trend')


  #dataset = dataset.dropna()
  print(dataset.head(60))
  
  quantile_list = [0, .25, .5, .75, 1.]
  quantiles = dataset['Volume'].quantile(quantile_list)
  #print("quantiles: ", quantiles)

  # fig, ax = plt.subplots()
  # dataset['Volume'].hist(bins=30, color='#A9C5D3', edgecolor='black', grid=False)
  # for quantile in quantiles:
  #   qvl = plt.axvline(quantile, color='r')
  #   ax.legend([qvl], ['Quantiles'], fontsize=10)
  #   ax.set_title('Volumn', fontsize=12)
  #   ax.set_xlabel('Volumn range', fontsize=12)
  #   ax.set_ylabel('Frequency', fontsize=12)

  #plt.hist(dataset['Volume'], bins = 30)
  #plt.xlabel("Number")
  #plt.ylabel("Frequency")
  #plt.show()

  quantile_labels = ['vol_low', 'vol_medium_low', 'vol_medium_high', 'vol_high']
  dataset['volumn_quantile_range'] = pd.qcut(
                                            dataset['Volume'], 
                                            q=quantile_list)
  dataset['volumn_quantile_label'] = pd.qcut(
                                            dataset['Volume'], 
                                            q=quantile_list,       
                                            labels=quantile_labels)

#################################
# Get previous close price trend
#################################
def check_prevclose_uptrend(d1, d2, d3, d4):
  dataset['prevclose_up_trend'] = np.logical_and(np.logical_and(d1>d4, d2>d4), d3>d4)

# next 3 days close price greater than sma3
def check_prevclose_downtrend(d1, d2, d3, d4):
  dataset['prevclose_down_trend'] = np.logical_and(np.logical_and(d1<d4, d2<d4), d3<d4)

def combined_prevclose_trends(d1, d2):
  dataset['test1'] = ((d1 == False) & (d2 == False)).apply(lambda x: 0 if x==True else 0)  
  dataset['test2'] = ((d1 == True) & (d2 == False)).apply(lambda x: 1 if x==True else 0)
  dataset['test3'] = ((d1 == False) & (d2 == True)).apply(lambda x: 2 if x==True else 0) 
  #dataset['prevclose_price_trend'] = (np.add(np.add(dataset['test1'], dataset['test2']), dataset['test3'])).apply(lambda x: 'prevclose_na' if x==0 else 'prevclose_up_trend' if x==1 else 'prevclose_down_trend')
  dataset['prevclose_price_trend'] = (np.add(np.add(dataset['test1'], dataset['test2']), dataset['test3']))

def getPrevCloseTrend(dataset):
  dataset['prevclose_1'] = dataset.Close.shift(1)
  dataset['prevclose_2'] = dataset.Close.shift(2)
  dataset['prevclose_3'] = dataset.Close.shift(3)
  dataset['prevclose_sma5'] = talib.SMA(dataset['prevclose_1'], timeperiod=5)  
  check_prevclose_uptrend(dataset['prevclose_1'], dataset['prevclose_2'], dataset['prevclose_3'], dataset['prevclose_sma5'])
  check_prevclose_downtrend(dataset['prevclose_1'], dataset['prevclose_2'], dataset['prevclose_3'], dataset['prevclose_sma5'])
  #dataset.dropna(inplace=True)
  combined_prevclose_trends(dataset['prevclose_up_trend'], dataset['prevclose_down_trend'])
  print(dataset.head(30))


###########################
# create outcome dataframe
###########################
# next 3 days close price greater than sma3
def check_uptrend(d1, d2, d3, d4):
  outcomes['up_trend'] = np.logical_and(np.logical_and(d1>d4, d2>d4), d3>d4)

# next 3 days close price greater than sma3
def check_downtrend(d1, d2, d3, d4):
  outcomes['down_trend'] = np.logical_and(np.logical_and(d1<d4, d2<d4), d3<d4)

# combine up_trend and down_trend, re-label for feature association 
def combined_trends(d1, d2):
  outcomes['test1'] = ((d1 == False) & (d2 == False)).apply(lambda x: 0 if x==True else 0)  
  outcomes['test2'] = ((d1 == True) & (d2 == False)).apply(lambda x: 1 if x==True else 0)
  outcomes['test3'] = ((d1 == False) & (d2 == True)).apply(lambda x: 2 if x==True else 0) 
  #outcomes['nextclose_price_trend'] = (np.add(np.add(outcomes['test1'], outcomes['test2']), outcomes['test3'])).apply(lambda x: 'nextclose_na' if x==0 else 'nextclose_up_trend' if x==1 else 'nextclose_down_trend')
  outcomes['nextclose_price_trend'] = (np.add(np.add(outcomes['test1'], outcomes['test2']), outcomes['test3']))

def createOutcomesDataframe(dataset): 
  # next day's opening change
  outcomes['Close'] = dataset.Close
  #outcomes['open_1'] = dataset.Open.shift(-1)/dataset.Close.shift(0)-1
  # next day's closing change
  #outcomes['close_1'] = dataset.Close.pct_change(-1)
  #outcomes['close_1_CAT'] = dataset.Close.pct_change(-1).apply(lambda d: 1 if d>=0 else -1)
  #outcomes['close_5'] = dataset.Close.pct_change(-5) 
  #outcomes['close_5'] = talib.ROC(dataset['Close'], timeperiod=5).shift(-5)
  #outcomes['close_5_CAT'] = outcomes['close_5'].apply(lambda d: 'close_up_5day' if d>=0 else 'close_down_5day')
  outcomes['close_1'] = dataset.Close.shift(-1)
  outcomes['close_2'] = dataset.Close.shift(-2)
  outcomes['close_3'] = dataset.Close.shift(-3)
  outcomes['close_4'] = dataset.Close.shift(-4)
  outcomes['close_5'] = dataset.Close.shift(-5)
  #outcomes['sma5'] = talib.SMA(outcomes['close_5'], timeperiod=5)  
  outcomes['sma5'] = (outcomes['close_1']+outcomes['close_2']+outcomes['close_3']+outcomes['close_4']+outcomes['close_5'])/5
  check_uptrend(outcomes['close_1'], outcomes['close_2'], outcomes['close_3'], outcomes['sma5'])
  check_downtrend(outcomes['close_1'], outcomes['close_2'], outcomes['close_3'], outcomes['sma5'])
  outcomes.dropna(inplace=True)
  combined_trends(outcomes['up_trend'], outcomes['down_trend'])
  #print(outcomes.isnull().values.any())
  #print(outcomes.tail(15))
  print(outcomes.tail(30))

def showHistogram(data):
  # show histogram of next day's closing change categories
  # An "interface" to matplotlib.axes.Axes.hist() method
  # n, bins, patches = plt.hist(x=data, bins='auto', color='#0504aa', alpha=0.7, rwidth=0.85)
  # plt.grid(axis='y', alpha=0.75)
  # plt.xlabel('Value Range')
  # plt.ylabel('Frequency')
  # plt.title("Next day's closing change categories")
  # plt.text(23, 45, r'$\mu=15, b=3$')
  # maxfreq = n.max()
  # # Set a clean upper y-axis limit.
  # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
  data.plot.hist(grid=True, bins=30, rwidth=0.9, color='#607c8e')
  plt.title("Next day's closing change categories")
  plt.xlabel('Counts')
  plt.ylabel('Value')
  plt.grid(axis='y', alpha=0.75)
  plt.show()


############################### main program ###############################

#########################
# read data source files
#########################
# read HSI data source
dataset = read_csv('data/HSI.csv',header=0,parse_dates=[0],date_parser=parser,index_col=0)
dataset = dataset.dropna()

# read candlesticks patterns
candlesticks = read_csv('data/candlesticks_patterns.csv',header=0,parse_dates=[0],date_parser=parser,index_col=0)

# define outcomes dataframe
outcomes = DataFrame(index=dataset.index)

# features dataset
#getCandleSticks(candlesticks) 
getRSIChange(dataset)
#getCloseChange(dataset)
# getRocOFCloseChange(dataset)
getSMA(dataset)
#getVolumnChange(dataset)
getPrevCloseTrend(dataset)
#encodingExmaple()

# outcomes dataset
createOutcomesDataframe(dataset)

# visualize data
#outcomes['close_1_CAT'] = pd.cut(outcomes['close_1'], 8)
#showHistogram(outcomes['close_1'])

# combine features dataset and outcomes dataset
y = outcomes.nextclose_price_trend
#X = dataset
X = dataset[[
  'Open','High','Low','Close','Adj Close','Volume',
  'RSI_5day_vs_prevday','smaDiff_5_14','prevclose_price_trend'
]].copy()
Xy = X.join(y).dropna()
y = Xy[y.name]
X = Xy[X.columns]
print(y.shape)
print(X.shape)

# print(Xy.head(20))

# print(Xy.columns)

# Visualize correslation between variables
# Correction Matrix Plot
# Convert the input into a 2D dictionary
df_copy = Xy[[
  'RSI_5day_vs_prevday','smaDiff_5_14','prevclose_price_trend','nextclose_price_trend'
]].copy()
df_copy2 = Xy[[
  'Open','Close','High','Low','Volume'
]].copy()
#corr = df_copy2.corr(method='pearson')
#print(corr)

#####
# Create the plot
#####
#plt.pcolormesh(df, edgecolors='black')
#plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
#plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)


#########################
# output to CSV and exit
########################
dataset = dataset.dropna()
Xy.to_csv('data/preprocessed_features_2.csv')
sys.exit()
