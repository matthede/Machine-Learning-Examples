import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import uniform
from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import PReLU
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

# read data

crime = pd.read_csv('syracuse_crime.csv', header=0, index_col=0)

print(crime.info())

# clean data up to include day of week, crime type, crime area level of crime, and 
# time of day

crime['Month'] = crime['DATE'].str[5:7]
crime['Day'] = crime['DATE'].str[0:10]
crime['Day'] = pd.to_datetime(crime['Day']).dt.weekday_name

TODreported = []

for x in crime['TIMESTART']:
    if x >= 0 and x <= 500:
        TODreported.append('Late Night')
    elif x >= 501 and x <= 1200:
        TODreported.append('Morning')
    elif x >= 1201 and x <= 1600:
        TODreported.append('Afternoon')
    elif x >= 1601 and x <= 2000:
        TODreported.append('Evening')
    else:
        TODreported.append('Night')

crime['TODreported'] = TODreported

TypeofCrime = []

for x in crime['Offense']:
    if x in ('SIMPLE ASSAULT','OFFN AGAINST FAMILY','SEX OFFENSES','KIDNAPPING','ROBBERY','AGGRAVATED ASSAULT','ARSON','MURDER'): 
        TypeofCrime.append(1)
    else:
        TypeofCrime.append(0)

crime['TypeofCrime'] = TypeofCrime



CrimeArea = []

CrimeAreaCount = crime.ADDRESS.value_counts(normalize=True)
print(CrimeAreaCount.sort_values)


for x in crime['ADDRESS']:
    if x in ('1 DESTINY USA DR','500 S  STATE ST','0 S  SALINA ST','400 JAMES ST',
             '300 CROSSETT ST','1800 E  FAYETTE ST','500 S  CLINTON ST','0 BUTTERNUT ST',
             '0 SOUTH AV','700 JAMES ST','100 DICKERSON ST','500 S  SALINA ST','300 PROSPECT AV',
             '1000 COL EILEEN COLLINS BLVD','600 ROBINSON ST','600 S  SALINA ST','100 SMITH LN',
             '1400 S  SALINA ST','100 RONEY LN','0 S  CLINTON ST','0 WALSH CIR','300 COURT ST',
             '900 JAMES ST','100 DABLON CT','400 S  SALINA ST','100 BALLANTYNE RD','800 PARK ST',
             '0 MERRIMAN AV','2100 E  FAYETTE ST','700 BUTTERNUT ST','700 FIRST NORTH ST','0 PARK ST',
             '300 SOUTH AV','100 ROBINSON ST','0 GIFFORD ST','900 ONONDAGA AV'):
                 CrimeArea.append('High')
    else:
        CrimeArea.append('Normal')

crime['CrimeAreaReportedLevel'] = CrimeArea

crime = crime.drop(['DATE', 'TIMESTART', 'TIMEEND', 'ADDRESS', 'Offense'], axis=1)


# get variables ready for creating model

X = crime[['Month', 'TODreported', 'Day', 'CrimeAreaReportedLevel']]
X = pd.get_dummies(X)
y = crime['TypeofCrime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 123)



# Neural network model in sklearn

def nn(X, y):
    hidden_layer_sizes = [(100, 100, 100), (50, 50, 50), (50, 100, 50)]
    activation = ['tanh', 'logistic', 'identity', 'relu']
    solver = ['adam', 'sgd']
    alpha = uniform(.0001, 1)
    learning_rate = ['constant', 'adaptive', 'invscaling']
    mlp_param = dict(hidden_layer_sizes = hidden_layer_sizes, activation = activation, solver = solver, alpha = alpha, learning_rate = learning_rate)
    mlp = MLPClassifier()
    random_mlp = RandomizedSearchCV(mlp, mlp_param, cv = 5, n_iter= 100, random_state= 1)
    random_mlp.fit(X, y)
    print('Best learning_rate:', random_mlp.best_estimator_.get_params()['learning_rate'])
    print('Best activation:', random_mlp.best_estimator_.get_params()['activation'])
    print('Best solver:', random_mlp.best_estimator_.get_params()['solver'])
    print('Best alpha:', random_mlp.best_estimator_.get_params()['alpha'])
    print('hidden_layer_sizes:', random_mlp.best_estimator_.get_params()['hidden_layer_sizes'])


nn(X_train, y_train)

nn = MLPClassifier(activation='identity', learning_rate='adaptive', alpha= 0.18521430600649502, early_stopping= True, hidden_layer_sizes=(100, 100, 100), solver='adam', max_iter=500)

nn.fit(X_train, y_train)

prediction = nn.predict(X_test)

nn_score = accuracy_score(prediction, y_test)

print(nn_score)

# keras

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
columns = X.shape[1]
model.add(Dense(100, activation = 'relu', input_shape = (columns,)))
model.add(LeakyReLU(alpha=.01))
model.add(Dense(100, activation = 'relu', input_shape = (columns,)))
model.add(LeakyReLU(alpha=.01))
model.add(Dense(100, activation = 'relu', input_shape = (columns,)))
model.add(LeakyReLU(alpha=.01))
model.add(Dense(2, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
early_stopping_monitor = EarlyStopping(patience = 5)
model.fit(X_train, y_train, epochs = 1000, validation_split = 0.3, callbacks = [early_stopping_monitor])
predictions = model.predict(X_test)
predictions = predictions.argmax(1)
y_test = y_test.argmax(1)
score = accuracy_score(predictions, y_test)
print(score)