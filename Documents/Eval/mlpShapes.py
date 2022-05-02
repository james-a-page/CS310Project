from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tpot.builtins import StackingEstimator
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import time
import seaborn as sns

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
dataset = pd.read_csv('../Data/CurrentDataset.csv')
dataset['windgust'].replace(np.NaN, 0, inplace=True)
dataset = dataset.dropna().drop('Unnamed: 0',axis=1)
# dataset = dataset[dataset['loadFactor']<=1]
print(dataset.shape)

#Removing ['CLDCW-1' 'CLDNW-1' 'KILBW-1' 'MILWW-1' 'WHIHW-1' 'BLLA-1' 'FAARW-2' 'COUWW-1' 'FAARW-1'] from dataset as they have reported loadfactor > 1 => issue with location data
dataset = dataset[~dataset['BMUID'].isin([
    'CLDCW-1',
    'CLDNW-1',
    'KILBW-1',
    'MILWW-1',
    'WHIHW-1',
    'BLLA-1',
    'FAARW-2',
    'COUWW-1',
    'FAARW-1',
])]
print(dataset.shape)



y = dataset['loadFactor']
X = dataset.drop('BMUID',axis=1).drop('loadFactor',axis=1)
training_features, testing_features, training_target, testing_target = \
    train_test_split(X, y, train_size=0.80, random_state=42)


sizes = np.linspace(0.1,0.8,16)
shapes = [(25,), (50,), (100,), (50, 100), (25, 50, 100), (50, 100, 250), (100, 50, 100),(100,25,25,100), (100, 50, 25, 50, 100),(25, 50, 100, 50, 25), ]
scores = []
mses = []

training_features, testing_features, training_target, testing_target = \
                train_test_split(X, y,train_size=0.8,test_size=0.2,random_state=42)
scaler = StandardScaler() 
for i,x in enumerate(shapes):
    model = MLPRegressor(hidden_layer_sizes=x)
    pipeline = Pipeline(steps=[('preprocessor', scaler), ('classifier',
                                                          model)])
    pipeline.fit(training_features,training_target)
    results = pipeline.predict(testing_features)
    scores.append(pipeline.score(testing_features, testing_target))
    kfold = KFold(n_splits=5)
    cv_results = cross_val_score(pipeline,
                                 training_features,
                                 training_target,
                                 cv=kfold,
                                 scoring='neg_mean_squared_error')
    print("%s: %f (%f)" % (x, cv_results.mean(),scores[i]))

