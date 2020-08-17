import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


#read data in and prep by position.  Different sets can be used for model.

nfl = pd.read_csv('nfl_draft.csv', header=0, index_col=0)

print(nfl.info())

nfl['Year'] = nfl['Year'].astype(object)

nfl = nfl.loc[nfl['Year'].isin([2010, 2011, 2012, 2013, 2014])]

# QB

qb = nfl.loc[nfl['Pos']=='QB']

# RB

qb = nfl.loc[nfl['Pos']=='RB']

# WR 

wr = nfl.loc[nfl['Pos']=='WR']

# Defense

defense = nfl.loc[nfl['Pos'].isin(['DT', 'DE', 'OLB'])]

# visualize distribution of games played for all players and positions

sns.distplot(nfl['G'].dropna(),
             hist=False,
             kde_kws={'shade':True})
plt.show

# linear model plot of games played and draft position

sns.lmplot(data=nfl,
           x="Pick",
           y="G",
           row = 'Year')

#negative linear relationship shows a trend of later picks playing fewer games
# correlation plot to determine features for pro bowl prediction.csv

sns.heatmap(nfl.corr())
plt.show()

# model to classify whether a draft pick becomes a pro bowl player
# creating binary classification column for pro bowl
# random forest
 
nfl['PBplayer'] = np.where(nfl['PB']>=1, 1, 0)

model_data = nfl[['St', 'DrAV', 'G', 'PBplayer']]
model_data = model_data.dropna()

X = model_data[['St', 'DrAV', 'G']]
y = model_data['PBplayer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 123)

# create random search model

def model_rf(X, y):
    rfc = RandomForestClassifier()
    n_estimators = [20, 40, 60, 80, 200, 400, 600, 800, 1000]
    max_depth = [int(x) for x in np.linspace(10, 1000, num = 50)]
    max_depth.append(None)
    max_features = ['auto', 'sqrt']
    min_samples_split = [2, 4, 6, 8, 10]
    min_samples_leaf = [2, 4, 6, 8, 10]
    bootstrap = [True, False]
    rfparam = dict(n_estimators = n_estimators, max_depth = max_depth, max_features = max_features,
               min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, bootstrap = bootstrap) 
    randomrf = RandomizedSearchCV(rfc, rfparam, cv = 5, n_iter= 100, random_state= 1)
    rf_model = randomrf.fit(X, y)
    print('Best n_estimators:', rf_model.best_estimator_.get_params()['n_estimators'])
    print('Best max_features:', rf_model.best_estimator_.get_params()['max_features'])
    print('Best max_depth:', rf_model.best_estimator_.get_params()['max_depth'])
    print('Best min_samples_split:', rf_model.best_estimator_.get_params()['min_samples_split'])
    print('Best min_samples_leaf:', rf_model.best_estimator_.get_params()['min_samples_leaf'])
    print('Best bootstrap:', rf_model.best_estimator_.get_params()['bootstrap'])

model_rf(X_train, y_train)

rf = RandomForestClassifier(n_estimators= 400,max_features='auto',max_depth=979,
                            min_samples_split=6,min_samples_leaf=8,bootstrap=True)
                            
rf.fit(X_train, y_train)

prediction = rf.predict(X_test)

score = accuracy_score(prediction, y_test)

print("The random forest model is", score, "accurate.")