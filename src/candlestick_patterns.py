#####################################################################
# 1. Detect candlesticks patterns
# 2. Append to data file and stored as candlesticks_patterns.csv
#####################################################################

from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
import numpy as np
import pandas as pd
import talib
import sys

# parsing date function
def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')

# read csv file using specified parsing function and index column is the first column (index=0)
dataset = read_csv('data/HSI.csv',header=0,parse_dates=[0],date_parser=parser,index_col=0)

# remove NaN values, drop 'Adj Close' column 
dataset = dataset.dropna().drop(['Adj Close'], axis=1)

# preview the dataset
#print(dataset.head(n=10))
#print("Dataset shape: ", dataset.shape)

# set arrays of open, high, low and close prices
pOpen = dataset['Open'].values
pHigh = dataset['High'].values
pLow = dataset['Low'].values
pClose = dataset['Close'].values
volume = dataset['Volume'].values

# make a copy of the dataset
df = dataset.copy()

# Two Crows (-100 denotes bearish)
patternName = 'CDL2CROWS'
output = talib.CDL2CROWS(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Three Black Crows (-100 denotes bearish)
patternName = 'CDL3BLACKCROWS'
output = talib.CDL3BLACKCROWS(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Three Inside Up/Down (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDL3INSIDE'
output = talib.CDL3INSIDE(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Three-Line Strike (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDL3LINESTRIKE'
output = talib.CDL3LINESTRIKE(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Three Outside Up/Down (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDL3OUTSIDE'
output = talib.CDL3OUTSIDE(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Three Stars In The South (+100 denotes bullish)
patternName = 'CDL3STARSINSOUTH'
output = talib.CDL3STARSINSOUTH(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Three Advancing White Soldiers (+100 denotes bullish)
patternName = 'CDL3WHITESOLDIERS'
output = talib.CDL3WHITESOLDIERS(pOpen, pHigh, pLow, pClose)
df[patternName] = output 
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Abandoned Baby (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLABANDONEDBABY'
output = talib.CDLABANDONEDBABY(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Advance Block (-100 denotes bearish)
patternName = 'CDLADVANCEBLOCK'
output = talib.CDLADVANCEBLOCK(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Belt-hold (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLBELTHOLD'
output = talib.CDLBELTHOLD(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Breakaway (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLBREAKAWAY'
output = talib.CDLBREAKAWAY(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Closing Marubozu (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLCLOSINGMARUBOZU'
output = talib.CDLCLOSINGMARUBOZU(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Concealing Baby Swallow (+100 denotes bullish)
patternName = 'CDLCONCEALBABYSWALL'
output = talib.CDLCONCEALBABYSWALL(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Counterattack (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLCOUNTERATTACK'
output = talib.CDLCOUNTERATTACK(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Dark Cloud Cover (-100 denotes bearish)
patternName = 'CDLDARKCLOUDCOVER'
output = talib.CDLDARKCLOUDCOVER(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Doji (+100 denotes bullish)
patternName = 'CDLDOJI'
output = talib.CDLDOJI(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Doji Star (-100 denotes bearlish, +100 denotes bullish)
#print(patternName, ": \n", df[df[patternName] != 0][p
patternName = 'CDLDOJISTAR'
output = talib.CDLDOJISTAR(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Dragonfly Doji (+100 denotes bullish)
patternName = 'CDLDRAGONFLYDOJI'
output = talib.CDLDRAGONFLYDOJI(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Engulfing Pattern (-100 denotes bearlish, +100 denotes bullish)
patternName = 'CDLENGULFING'
output = talib.CDLENGULFING(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Evening Doji Star (-100 denotes bearlish)
patternName = 'CDLEVENINGDOJISTAR'
output = talib.CDLEVENINGDOJISTAR(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Evening Star (-100 denotes bearlish)
patternName = 'CDLEVENINGSTAR'
output = talib.CDLEVENINGSTAR(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Up/Down-gap side-by-side white lines (-100 denotes bearlish, +100 denotes bullish)
patternName = 'CDLGAPSIDESIDEWHITE'
output = talib.CDLGAPSIDESIDEWHITE(pOpen, pHigh, pLow, pClose)
df[patternName] = output 
#print(patternName, ": \n", df[df[patternName] != 0][patternName]) 

# Gravestone Doji (+100 denotes bullish)
patternName = 'CDLGRAVESTONEDOJI'
output = talib.CDLGRAVESTONEDOJI(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Hammer (+100 denotes bullish)
patternName = 'CDLHAMMER'
output = talib.CDLHAMMER(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Hanging Man (-100 denotes bearish)
patternName = 'CDLHANGINGMAN'
output = talib.CDLHANGINGMAN(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Harami Pattern (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLHARAMI'
output = talib.CDLHARAMI(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Harami Cross Pattern (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLHARAMICROSS'
output = talib.CDLHARAMICROSS(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# High-Wave Candle (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLHIGHWAVE'
output = talib.CDLHIGHWAVE(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Hikkake Pattern
patternName = 'CDLHIKKAKE'
output = talib.CDLHIKKAKE(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Modified Hikkake Pattern
patternName = 'CDLHIKKAKEMOD'
output = talib.CDLHIKKAKEMOD(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Homing Pigeon (+100 denotes bullish)
patternName = 'CDLHOMINGPIGEON'
output = talib.CDLHOMINGPIGEON(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Identical Three Crows (-100 denotes bearish)
patternName = 'CDLIDENTICAL3CROWS'
output = talib.CDLIDENTICAL3CROWS(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# In-Neck Pattern (-100 denotes bearish)
patternName = 'CDLINNECK'
output = talib.CDLINNECK(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Inverted Hammer (+100 denotes bullish)
patternName = 'CDLINVERTEDHAMMER'
output = talib.CDLINVERTEDHAMMER(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Kicking (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLKICKING'
output = talib.CDLKICKING(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Kicking - bull/bear determined by the longer marubozu (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLKICKINGBYLENGTH'
output = talib.CDLKICKINGBYLENGTH(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Ladder Bottom (+100 denotes bullish)
patternName = 'CDLLADDERBOTTOM'
output = talib.CDLLADDERBOTTOM(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Long Legged Doji (+100 denotes bullish)
patternName = 'CDLLONGLEGGEDDOJI'
output = talib.CDLLONGLEGGEDDOJI(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Long Line Candle (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLLONGLINE'
output = talib.CDLLONGLINE(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Marubozu (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLMARUBOZU'
output = talib.CDLMARUBOZU(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Matching Low (+100 denotes bullish)
patternName = 'CDLMATCHINGLOW'
output = talib.CDLMATCHINGLOW(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Mat Hold (+100 denotes bullish)
patternName = 'CDLMATHOLD'
output = talib.CDLMATHOLD(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Morning Doji Star (+100 denotes bullish)
patternName = 'CDLMORNINGDOJISTAR'
output = talib.CDLMORNINGDOJISTAR(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Morning Star (+100 denotes bullish)
patternName = 'CDLMORNINGSTAR'
output = talib.CDLMORNINGSTAR(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# On-Neck Pattern (-100 denotes bearish)
patternName = 'CDLONNECK'
output = talib.CDLONNECK(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Piercing Pattern (+100 denotes bullish)
patternName = 'CDLPIERCING'
output = talib.CDLPIERCING(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Rickshaw Man (+100 denotes bullish)
patternName = 'CDLRICKSHAWMAN'
output = talib.CDLRICKSHAWMAN(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Rising/Falling Three Methods (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLRISEFALL3METHODS'
output = talib.CDLRISEFALL3METHODS(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Separating Lines (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLSEPARATINGLINES'
output = talib.CDLSEPARATINGLINES(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])
 
# Shooting Star (-100 denotes bearish)
patternName = 'CDLSHOOTINGSTAR'
output = talib.CDLSHOOTINGSTAR(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Short Line Candle (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLSHORTLINE'
output = talib.CDLSHORTLINE(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Spinning Top (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLSPINNINGTOP'
output = talib.CDLSPINNINGTOP(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Stalled Pattern (-100 denotes bearish)
patternName = 'CDLSTALLEDPATTERN'
output = talib.CDLSTALLEDPATTERN(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Stick Sandwich (-100 denotes bearish)
patternName = 'CDLSTICKSANDWICH'
output = talib.CDLSTICKSANDWICH(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Takuri (Dragonfly Doji with very long lower shadow) (+100 denotes bullish)
patternName = 'CDLTAKURI'
output = talib.CDLTAKURI(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Tasuki Gap (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLTASUKIGAP'
output = talib.CDLTASUKIGAP(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Thrusting Pattern (-100 denotes bearish)
patternName = 'CDLTHRUSTING'
output = talib.CDLTHRUSTING(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Tristar Pattern (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLTRISTAR'
output = talib.CDLTRISTAR(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])
 
# Unique 3 River (+100 denotes bullish)
patternName = 'CDLUNIQUE3RIVER'
output = talib.CDLUNIQUE3RIVER(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Upside Gap Two Crows (-100 denotes bearish)
patternName = 'CDLUPSIDEGAP2CROWS'
output = talib.CDLUPSIDEGAP2CROWS(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

# Upside/Downside Gap Three Methods (-100 denotes bearish, +100 denotes bullish)
patternName = 'CDLXSIDEGAP3METHODS'
output = talib.CDLXSIDEGAP3METHODS(pOpen, pHigh, pLow, pClose)
df[patternName] = output
#print(patternName, ": \n", df[df[patternName] != 0][patternName])

df.to_csv('data/candlesticks_patterns.csv')
sys.exit()
