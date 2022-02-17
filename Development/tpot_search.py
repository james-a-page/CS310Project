from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

dataset = pd.read_csv('../Data/CurrentDataset.csv')


dataset['wgust'].replace('', np.nan, inplace=True)
dataset = dataset.dropna().drop('Unnamed: 0',axis=1)
dataset = dataset[dataset['loadFactor']<=1]
y = dataset['loadFactor']
X = dataset.drop('BMUID',axis=1).drop('loadFactor',axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25)

tpot = TPOTRegressor(generations=100, population_size=100, verbosity=2,scoring='neg_mean_absolute_error',n_jobs=-1)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_exported_pipeline.py')