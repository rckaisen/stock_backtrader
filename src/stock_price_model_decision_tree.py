import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from pandas import read_csv
from pandas import datetime
from pandas import concat
from pandas import DataFrame
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# parsing date
def parser(x):
  return datetime.strptime(x,'%Y-%m-%d')

def image_path(fig_id):
  PROJECT_ROOT_DIR = "."  
  return os.path.join(PROJECT_ROOT_DIR, "data", fig_id)

# main
dataset = read_csv('data/preprocessed_features.csv',header=0,parse_dates=[0],date_parser=parser,index_col=None)

df = dataset[[
  'RSI_5day_vs_prevday','smaDiff_5_14','prevclose_price_trend','nextclose_price_trend'
]].copy()

le = preprocessing.LabelEncoder()
df = df.apply(le.fit_transform)

X = df.values[:, 0:3]
y = df.values[:,-1]

tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
tree_clf.fit(X, y)

print(tree_clf)

export_graphviz(
  tree_clf,
  out_file=image_path("stock_prediction_tree.dot"),
  feature_names=df.columns.values[0:3],
  class_names=['down','na', 'up'],
  rounded=True,
  filled=True
)