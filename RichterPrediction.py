import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from lightgbm import LGBMClassifier

from scipy.stats import uniform

train = pd.read_csv('train.csv', header = 0, index_col= 0)
test = pd.read_csv('test.csv', header = 0, index_col= 0)

train.info()
test.info()

# get dummy variables

train = pd.get_dummies(train)
test = pd.get_dummies(test)

# create set of dependent variables and independent variables and scale.

X = train.drop(['damage_grade'], axis = 1)
y = train['damage_grade']

scaler = StandardScaler()
X = scaler.fit_transform(X)
test = scaler.fit_transform(test)

# split to test out for f1 score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 123)

# add column names to x_train and x_test for feature selection process

column_names = train.drop(['damage_grade'], axis = 1)
X_train = pd.DataFrame(X_train, columns = column_names.columns)
X_test = pd.DataFrame(X_test, columns = column_names.columns)
test = pd.DataFrame(test, columns = column_names.columns)


# use random forest for feature selection


rfc = RandomForestClassifier()

rfc.fit(X_train, y_train)

print(rfc.feature_importances_)

features = pd.Series(rfc.feature_importances_, index= X_train.columns)

features.nlargest(25).plot(kind = 'bar', color = 'rgby')
plt.show()

# Based on the feature importance, new data will be tested

X_train = X_train[['geo_level_1_id', 
                   'geo_level_2_id', 
                   'geo_level_3_id', 
                   'age', 
                   'area_percentage', 
                   'height_percentage', 
                   'count_families', 
                   'foundation_type_r', 
                   'count_floors_pre_eq', 
                   'has_superstructure_mud_mortar_stone',
                   'has_superstructure_timber',
                   'ground_floor_type_v',
                   'land_surface_condition_t',
                   'land_surface_condition_n',
                   'roof_type_n',
                   'other_floor_type_q']]
                  
X_test = X_test[['geo_level_1_id', 
                   'geo_level_2_id', 
                   'geo_level_3_id', 
                   'age', 
                   'area_percentage', 
                   'height_percentage', 
                   'count_families', 
                   'foundation_type_r', 
                   'count_floors_pre_eq', 
                   'has_superstructure_mud_mortar_stone',
                   'has_superstructure_timber',
                   'ground_floor_type_v',
                   'land_surface_condition_t',
                   'land_surface_condition_n',
                   'roof_type_n',
                   'other_floor_type_q']]
test = test[['geo_level_1_id', 
                   'geo_level_2_id', 
                   'geo_level_3_id', 
                   'age', 
                   'area_percentage', 
                   'height_percentage', 
                   'count_families', 
                   'foundation_type_r', 
                   'count_floors_pre_eq', 
                   'has_superstructure_mud_mortar_stone',
                   'has_superstructure_timber',
                   'ground_floor_type_v',
                   'land_surface_condition_t',
                   'land_surface_condition_n',
                   'roof_type_n',
                   'other_floor_type_q']]

# lgb

learning_rate = [.01, .05, .1, .5, .75, .9]
max_depth = [-1, 1, 3, 4, 5, 7, 9]
min_child_samples = [10, 15, 20, 25, 30]
min_child_weight = [.001, .01, .1, .5]
num_leaves = [25, 27, 29, 31, 33, 35, 37, 39, 41, 45, 50]

params = dict(learning_rate = learning_rate, max_depth = max_depth, min_child_samples = min_child_samples,
              min_child_weight = min_child_weight, num_leaves = num_leaves)
              

lgb = LGBMClassifier()              
randomlgb = RandomizedSearchCV(lgb, params, cv = 5, n_iter= 100, random_state = 1)


randomlgb.fit(X_train, y_train)

print(randomlgb.best_estimator_.get_params())

lgb = LGBMClassifier(max_depth=7, learning_rate=.05, min_child_samples=10, 
                    min_child_weight=.1, num_leaves=50, n_estimators = 10000)

lgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds = 50, eval_metric = 'logloss')

predictions = lgb.predict(test)

print(predictions)

np.savetxt('C:/Users/David/Desktop/predictions.csv', predictions, delimiter=',')