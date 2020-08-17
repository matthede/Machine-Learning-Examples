import numpy as np
import pandas as pd
import matplotlib as mp
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from scipy.stats import uniform
from sklearn.metrics import accuracy_score

# Read data in and clean up

reviews = pd.read_csv("googleplaystore_user_reviews.csv", index_col= 0, header = 0)

print(reviews.info())

data = reviews[['Translated_Review', 'Sentiment']]

data = data.dropna()

cht = data['Sentiment'].value_counts().plot(kind = 'bar',
                                            title = "Sentiment Totals")
cht.set_xlabel("Sentiment")
cht.set_ylabel("Total")

# Based on uneven distribution, train test to be stratified


X = data['Translated_Review']
y = data['Sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, 
stratify = y, random_state = 123)

tfidf = TfidfVectorizer(stop_words = 'english', max_df = .6)

tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

# Build algorithm

C = [.001, .01, .05, .1, .5, 1, 10, 100]
penalty = ['l1', 'l2']
param_grid = dict(C = C, penalty = penalty)

lr = LogisticRegression()
    
random_search = GridSearchCV(lr, param_grid, cv = 5)
model = random_search.fit(tfidf_train, y_train)

# Check best parameters

print("Best C:", model.best_estimator_.get_params()['C'])
print("Best penalty:", model.best_estimator_.get_params()['penalty'])

# predict

lr_predict = model.predict(tfidf_test)

# score

accuracy = accuracy_score(lr_predict, y_test)



print("Model Accuracy:", accuracy)