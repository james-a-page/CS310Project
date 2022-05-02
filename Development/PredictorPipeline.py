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
import time

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
            train_test_split(X, y,train_size=0.80,random_state=42)

# 1: Average CV score on the training set was: -0.16625557325887264
# exported_pipeline = make_pipeline(
#     MinMaxScaler(),
#     StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=2, min_samples_split=5, n_estimators=100)),
#     StackingEstimator(estimator=KNeighborsRegressor(n_neighbors=37, p=1, weights="distance")),
#     StandardScaler(),
#     KNeighborsRegressor(n_neighbors=15, p=1, weights="distance")
# )

# 2: Average CV score on the training set was: -0.16026048399327827
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=1, min_samples_split=5, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=9, min_samples_split=8, n_estimators=100)),
    VarianceThreshold(threshold=0.01),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.001, loss="linear", n_estimators=100)),
    SelectPercentile(score_func=f_regression, percentile=28),
    StackingEstimator(estimator=LinearSVR(C=0.5, dual=False, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.01)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.01, loss="absolute_error", max_depth=7, max_features=1.0, min_samples_leaf=9, min_samples_split=10, n_estimators=100, subsample=0.5)),
    StackingEstimator(estimator=LinearSVR(C=20.0, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=1e-05)),
    KNeighborsRegressor(n_neighbors=61, p=2, weights="distance")
)



exported_pipeline.fit(training_features, training_target)
starttime = time.perf_counter_ns()
results = exported_pipeline.predict(testing_features)
endtime = time.perf_counter_ns()
print(((endtime-starttime)/1000000000)/len(testing_target))
plt.scatter(testing_target,results,marker='x')
plt.show()
print(exported_pipeline.score(testing_features,testing_target))
print(metrics.mean_squared_error(testing_target,results))
print(((results-testing_target)).abs().mean(),(results-testing_target).abs().std())
