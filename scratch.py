import pandas as pd
import numpy as np
out = []
allocation_to_location = pd.read_csv('Data//locations.csv').drop(
        'capacity', axis=1)
for i in range(56):
    dataset = pd.read_csv('Data/TimeSeriesOfLocations/' +(allocation_to_location.iloc[i].BMU_ID).replace('-', '_') +'.csv',parse_dates=["datetime"])
    distributions = {}
    for feature in [
                        'temp', 'dew', 'humidity', 'precip', 'windgust',
                        'windspeed', 'sealevelpressure', 'cloudcover',
                        'visibility'
    ]:
                    #1 - Smooth Data
        featureData = dataset[[feature]]
        smoothedData = (featureData.ewm(alpha=0.5)).mean()
                    #2 - Take Mean & std of each smoothed weather feature
                    #3 - Sample 1000 number of data points from normal distribution defined by step 2 (for each feature)
        distributions[feature] = (smoothedData.mean(),smoothedData.std())
        out += distributions
                #4 - Predict output over those points.
                # predictions = pred.predict(pd.DataFrame(distributions), model)
                # predictionList = np.append(predictionList, predictions)
                # predictions.mean()  #Mean of all predictions in this location
                # total += (predictions.mean() * count
print(out[0])