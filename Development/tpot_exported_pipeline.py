import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, f_regression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: -0.16026048399327827
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=1, min_samples_split=5, n_estimators=100)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.55, min_samples_leaf=9, min_samples_split=8, n_estimators=100)),
    VarianceThreshold(threshold=0.01),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.001, loss="linear", n_estimators=100)),
    SelectPercentile(score_func=f_regression, percentile=28),
    StackingEstimator(estimator=LinearSVR(C=0.5, dual=False, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.01)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.95, learning_rate=0.01, loss="lad", max_depth=7, max_features=1.0, min_samples_leaf=9, min_samples_split=10, n_estimators=100, subsample=0.5)),
    StackingEstimator(estimator=LinearSVR(C=20.0, dual=True, epsilon=0.001, loss="epsilon_insensitive", tol=1e-05)),
    KNeighborsRegressor(n_neighbors=61, p=2, weights="distance")
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
