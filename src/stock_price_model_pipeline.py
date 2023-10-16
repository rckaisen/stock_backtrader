######## example 1 ##########
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# add your data here
X_train,y_train = make_my_dataset()

# it takes a list of tuples as parameter
pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('clf', LogisticRegression())
])

# use the pipeline object as you would
# a regular classifier
pipeline.fit(X_train,y_train)

######## example 2 ##########
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

# X_train and X_test are lists of strings, each 
# representing one document
# y_train and y_test are vectors of labels
X_train,X_test,y_train,y_test = make_my_dataset()

# this calculates a vector of term frequencies for 
# each document
vect = CountVectorizer()

# this normalizes each term frequency by the 
# number of documents having that term
tfidf = TfidfTransformer()

# this is a linear SVM classifier
clf = LinearSVC()

pipeline = Pipeline([
    ('vect',vect),
    ('tfidf',tfidf),
    ('clf',clf)
])

# call fit as you would on any classifier
pipeline.fit(X_train,y_train)

# predict test instances
y_preds = pipeline.predict(X_test)

# calculate f1
mean_f1 = f1_score(y_test, y_preds, average='micro')