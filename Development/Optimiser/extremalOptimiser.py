from string import printable
import model as pred
import pandas as pd
import numpy as np
import math

global allocation_to_location
# global location_to_expected_weather

class Chromosome:
    def __init__(self,size,budget):
        self.size = size
        self.budget = budget
        #Generate inital random distribution that sums to less than the budget
        self.genes = (np.random.dirichlet(np.ones(self.size),size=1)[0]) * self.budget
        self.genes = [math.floor(x) for x in self.genes]

class Generation:
    def __init__(self,genNumber,size,parameters,fitnessModel,population=[]):
        self.genNumber = genNumber
        self.popSize = size
        self.geneCount,self.budget = parameters
        if population:
            self.population = population
        else:
            self.population = [Chromosome(self.geneCount,self.budget) for x in range(self.popSize)]
    
    def getPopulation(self):
        return [c.genes for c in self.population]

    #https://link.springer.com/content/pdf/10.1007%2F978-1-4614-6940-7.pdf - Elitist Non-dominated Sorting GA (NSGA-II)
    def getOffspring(self):
        newPop = []
        return Generation(self.genNumber+1,self.popSize,(self.geneCount,self.budget),newPop)

def fitness(allocation,model):
    value = 0
    for i,count in enumerate(allocation):
        if count > 0:
            # print(allocation_to_location.iloc[i].BMU_ID)
            dataset = pd.read_csv('../../Data/TimeSeriesOfLocations/'+(allocation_to_location.iloc[i].BMU_ID).replace('-','_')+'.csv',parse_dates=["datetime"])
            distributions = {}
            for feature in ['temp','dew','humidity','precip','windgust','windspeed','sealevelpressure','cloudcover','visibility']:
                #1 - Smooth Data
                featureData = dataset[[feature]]
                smoothedData = (featureData.ewm(alpha=0.1)).mean()
                #2 - Take Mean & std of each smoothed weather feature
                #3 - Sample _ number of data points from guassian distribution defined by step 2 (for each feature)
                distributions[feature] = np.random.normal(smoothedData.mean(),smoothedData.std(),10000)
            #4 - Predict output over those points.
            predictions = pred.predict(pd.DataFrame(distributions),model)
            value += round(predictions.mean(),5) * count
        # F1 = mean of all allocations
        # F2 = Lower quartile of all locations
        # F3 = ?
        # print(np.quantile(predictions,0.25))
        # print(np.quantile(predictions,0.5))
        # print(np.quantile(predictions,0.75))
        # pred.predict([],model) 
    return value

#Based upon: https://www.sciencedirect.com/science/article/pii/S0020025515007276
def main():
    global allocation_to_location
    # global location_to_expected_weather
    allocation_to_location = pd.read_csv('../../Data//locations.csv').drop('capacity',axis=1)
    predictor = pred.initaliseModel()
    chromosomeParameters = (56,120)
    gen0 = Generation(0,1,chromosomeParameters,predictor)
    print(gen0.getPopulation())
    print(fitness(gen0.getPopulation()[0],predictor))




if __name__ == "__main__":
    main()