#####################################################
# Examine the association between features
#
# Formula to calculate number of association rules R
# given number of items d:
#    R = 3^d - 2^(d+1) +1
#####################################################

########################
# --- Features: ---
# 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'candlesticks',
# 'RSI', 'RSI_PREV', 'RSI_PREV_CAT', 'RSI_AVG_5DAYS', 'RSI_AVG_5DAYS_CAT',
# 'RSI_5day_vs_prevday', 'sma5', 'sma14', 'sma50', 'sma200',
# 'smaDiff_5_14', 'smaDiff_5_50', 'smaDiff_5_200', 'smaDiff_14_50',
# 'smaDiff_14_200', 'smaDiff_50_200', 'OBV', 'Volume_CHG',
# 'volumn_quantile_range', 'volumn_quantile_label', 'price_trend'
########################

########################
# Current dataset:
#   from 2001-Aug to 2018-Oct
#   ie. 206 months
########################

from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from apyori import apriori
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt  
import sys

# parsing date
def parser(x):
  return datetime.strptime(x,'%Y-%m-%d')

# Class labeling  
def labeling_candlesticks(val):
  if (val == -3):
    return 'candle_extremely_bearish'
  elif (val == -2):
    return 'candle_strongly_bearish'
  elif (val == -1):
    return 'candle_bearish'
  elif (val == 0):
    return 'candle_neutral'
  elif (val == 1):
    return 'candle_bullish'
  elif (val == 2):
    return 'candle_strongly_bullish'
  elif (val == 3):
    return 'candle_extremely_bullish'

def labeling_rsi(val):
  if (val == 0): 
    return 'rsi_down'
  elif (val == 1):
    return 'rsi_up'

def labeling_close_change(val):
  if (val == 0):
    return 'closeChg_down'
  elif (val == 1):
    return 'closeChg_up'

def labeling_roc_close_change(val):
  if (val == -1):
    return 'rocCloseChg_slowDown'
  elif (val == 1):
    return 'rocCloseChg_speedUp'

def labeling_volume_change(val):
  if (val < -1):
    return 'v1'
  elif (val >= -1 and val < 0):
    return 'v2'
  elif (val >= 0 and val < 1):
    return 'v3'
  elif (val >= 1 and val < 2):
    return 'v4'
  elif (val >= 2 and val < 3):
    return 'v5'
  elif (val >= 3 and val < 4):
    return 'v6'
  elif (val >= 4 and val < 5):
    return 'v7'
  elif (val >= 5 and val < 6):
    return 'v8'
  elif (val >= 7):
    return 'v9'  

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

# read data file
dataset = read_csv('data/preprocessed_features.csv',header=0,parse_dates=[0],date_parser=parser,index_col=0)
#print(dataset.columns)
#print(dataset.values)

# re-labeling of data value
df = dataset[[
  'RSI_5day_vs_prevday','smaDiff_5_14','prevclose_price_trend','nextclose_price_trend'
]].copy()
#df['candlesticks'] = df['candlesticks'].apply(lambda v: labeling_candlesticks(v))
#df['RSI_CHG_CAT'] = df['RSI_CHG_CAT'].apply(lambda v: labeling_rsi(v))
#df['rocOFCloseChangeCat'] = df['rocOFCloseChangeCat'].apply(lambda v: labeling_roc_close_change(v))
#df['RSI_CHG_ROC_CAT'] = pd.cut(df['RSI_CHG_ROC'].replace(np.inf, np.nan).dropna(), 10)
#df['Volume_CHG'] = df['Volume_CHG'].apply(lambda v: labeling_volume_change(v))
#df['CLOSE_CHG_CAT'] = df['CLOSE_CHG_CAT'].apply(lambda v: labeling_close_change(v))
#df['close_1_CAT'] = df['close_1_CAT'].apply(lambda v: 'close_up' if (v==1) else 'close_down')

#print(df.head(20))
#print(df.tail(20))

print(df.shape)
#print(df.values)

# find and apply association rules 
rowCount = df.shape[0]  # number of rows
itemCount = df.shape[1] # number of items

records = []  
for i in range(0, rowCount):  
    records.append([str(df.values[i,j]) for j in range(0, itemCount)])

#print(records[0])

min_support_setting = 0.0002  
min_confidence_setting = 0.5
min_lift_setting = 1.0
min_length_setting = 2

association_rules = list(apriori(records, min_support=min_support_setting, min_confidence=min_confidence_setting, min_lift=min_lift_setting, min_length=min_length_setting)) 
association_results = list(association_rules) 

#print("Count of association rules: ", len(association_rules))
#print(association_results)

#####################
# Write to text file
#####################
# file = open("data/association_rules_008.txt","w") 
 
# file.write("===== Parameters =====" + "\n")
# file.write("min_support: " + str(min_support_setting) + "\n")
# file.write("min_confidence: " + str(min_confidence_setting) + "\n")
# file.write("min_lift: " + str(min_lift_setting) + "\n")
# file.write("min_length_setting: " + str(min_length_setting) + "\n\n")
 
# file.write("Count of association rules: " + str(len(association_rules)) + "\n\n")

# count = 0
# for item in association_rules:
#   #print(item[0]) # forzenset of items
#   for element in item[2]:  # item[2] is ordered_statistics
#     if (len(element[0]) > 0 and ('up_trend' in element[1] or 'down_trend' in element[1])):
#       index = count + 1
#       file.write("--- Rule " + str(index) + " ---" + "\n")
#       file.write("Support: " + str(item[1]) + "\n") # support
#       file.write("Rule: " + str(element[0]) + " -> " + str(element[1]) + "\n")  # items_base -> items_add
#       file.write("Confidence: " + str(element[2]) + "\n")  # confidence
#       file.write("Lift: " + str(element[3]) + "\n")  # lift
#       file.write("=====================================" + "\n")
#       count += 1
    
# file.write("\n" + "Count of association rules: " + str(count) + "\n\n")

# file.close()

#####################
# Write to CSV file
#####################
# candlesticks
candlesticks_set = {
  "candle_extremely_bearish",
  "candle_strongly_bearish",
  "candle_bearish",
  "candle_neutral",
  "candle_bullish",
  "candle_strongly_bullish",
  "candle_extremely_bullish"
}
  
# RSI
rsi_set = {
  "noChange_below30",
  "noChange_btw30and70",
  "noChange_above70",
  "below30_to_btw30and70",
  "btw30and70_to_above70",
  "above70_to_btw30and70",
  "btw30and70_to_below30", 
  "below30_to_above70",
  "above70_to_below30"
}
  
# SMA
sma_set = {  
  "sma5_cross_above_sma14", 
  "sma5_cross_below_sma14",
  "sma5_below_sma14",
  "sma5_above_sma14"
}
  
# volume
volume_set = {
  "vol_up_trend",
  "vol_down_trend",
  "vol_normal"
}

# previous 3 days close price trend
prevclose_price_trend_set = {
  "prevclose_up_trend",
  "prevclose_down_trend",
  "prevclose_na"
}

# next 3 days close price trend
nextclose_price_trend_set = {
  "nextclose_up_trend",
  "nextclose_down_trend",
  "nextclose_na"
}

file = open("data/association_rules_001.csv","w") 
 
file.write("min_support,min_confidence,min_lift,RSI_5day_vs_prevday,smaDiff_5_14,prevclose_price_trend,nextclose_price_trend" + "\n")

for item in association_rules:
  #print(item[0]) # forzenset of items
  for element in item[2]:  # item[2] is ordered_statistics
    if (len(element[0]) > 0 and ('nextclose_up_trend' in element[1] or 'nextclose_down_trend' in element[1])):
      value1 = 'null'
      value2 = 'null'
      value3 = 'null'
      value4 = 'null'

      fs1 = element[0]
      for x in fs1:
        if (x in rsi_set):
          value1 = x
        elif (x in sma_set):
          value2 = x
        elif (x in prevclose_price_trend_set):
          value3 = x  
  
      fs2 = element[1]
      for y in fs2:
        if (y in nextclose_price_trend_set):
          value4 = y
    
      file.write(str(item[1]) + "," + 
                 str(element[2]) + "," + 
                 str(element[3]) + "," +
                 value1 + "," +
                 value2 + "," +
                 value3 + "," +
                 value4 + "," + "\n") 

file.close()

sys.exit()


