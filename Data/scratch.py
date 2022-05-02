from math import gcd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# dataset = pd.read_csv('../Data/CurrentDataset.csv')
# dataset['windgust'].replace(np.NaN, 0, inplace=True)
# dataset = dataset.dropna().drop('Unnamed: 0', axis=1)
# dataset = dataset[dataset['loadFactor']<=1]

# #Removing ['CLDCW-1' 'CLDNW-1' 'KILBW-1' 'MILWW-1' 'WHIHW-1' 'BLLA-1' 'FAARW-2' 'COUWW-1' 'FAARW-1'] from dataset as they have reported loadfactor > 1 => issue with location data
# dataset = dataset[~dataset['BMUID'].isin([
#     'CLDCW-1'
#     'CLDNW-1'
#     'KILBW-1'
#     'MILWW-1'
#     'WHIHW-1'
#     'BLLA-1'
#     'FAARW-2'
#     'COUWW-1'
#     'FAARW-1'
# ])]


# sns.set()
# sns.pairplot(dataset,y_vars='loadFactor',markers='x')
# plt.show()


loc = pd.read_csv('./locations.csv')
# print(loc.capacity)/
# print(np.lcm(loc.capacity.round().astype(int)))
a = [100, 200, 150]  # will work for an int array of any length
lcm = 1
for i in loc.capacity.round().astype(int):
    lcm = lcm*i//gcd(lcm, i)
print(lcm)
alloc = []
for i in loc.capacity.round().astype(int):
    alloc.append(lcm//(i*1000000000000000000000000))
print(alloc)

allocation_to_location = pd.read_csv('./locations.csv').drop(
    'capacity', axis=1)
predictionList = np.array([])
      # save = pd.DataFrame(data=[],columns=['BMU_ID','predOutput'])
precomputedPred = pd.read_csv('./PreComputedPredictions.csv')
       #For each location, model the expected weather, sample that distribution, predict outputs based on that sample
for i, count in enumerate(alloc):
    print(i)
    if count > 0:

        predictions = ((precomputedPred.loc[
                    precomputedPred['BMU_ID'] ==
                    allocation_to_location.iloc[i].BMU_ID]).predOutput).array
        for j in range(count):
            predictionList = np.append(predictionList, (predictions))

        # F1 = mean of all allocations (maxmise)
        #Total predicted output of this allocation as 'Load factor' percentage (% of total capacity allocated)
print(predictionList.mean())
print(np.var(predictionList))
print(predictionList.min())
print(predictionList.max())
