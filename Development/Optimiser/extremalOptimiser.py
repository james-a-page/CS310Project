from random import seed
from string import printable
import model as pred
import pandas as pd
import numpy as np
import math
from matplotlib import projections, pyplot as plt

global allocation_to_location
# global location_to_expected_weather


class Chromosome:

    def __init__(self, size, budget):
        self.size = size
        self.budget = budget
        #Generate inital random distribution that sums to less than the budget
        self.genes = (np.random.dirichlet(np.ones(self.size),
                                          size=1)[0]) * self.budget
        self.genes = [math.floor(x) for x in self.genes]

    def getGenes(self):
        return self.genes

    def fitness(self, model):

        #Check doesn't break constraints
        if sum(self.genes) > self.budget:
            return 0

        #Define Inital values
        total = 0
        predictionList = np.array([])
        # save = pd.DataFrame(data=[],columns=['BMU_ID','predOutput'])
        precomputedPred = pd.read_csv('../../Data/PreComputedPredictions.csv')
        #For each location, model the expected weather, sample that distribution, predict outputs based on that sample
        for i, count in enumerate(self.genes):
            np.random.seed(0)
            if count > 0:

                #Uncomment to recompute predictions at run time -- will be very slow on large runs
                # dataset = pd.read_csv(
                #     '../../Data/TimeSeriesOfLocations/' +
                #     (allocation_to_location.iloc[i].BMU_ID).replace('-', '_') +
                #     '.csv',
                #     parse_dates=["datetime"])
                # distributions = {}
                # for feature in [
                #         'temp', 'dew', 'humidity', 'precip', 'windgust',
                #         'windspeed', 'sealevelpressure', 'cloudcover',
                #         'visibility'
                # ]:
                #     #1 - Smooth Data
                #     featureData = dataset[[feature]]
                #     smoothedData = (featureData.ewm(alpha=0.5)).mean()
                #     #2 - Take Mean & std of each smoothed weather feature
                #     #3 - Sample 1000 number of data points from normal distribution defined by step 2 (for each feature)
                #     distributions[feature] = np.random.normal(
                #         smoothedData.mean(), smoothedData.std(), 1000)
                # #4 - Predict output over those points.
                # predictions = pred.predict(pd.DataFrame(distributions), model)
                # save = pd.concat([save,pd.DataFrame([(allocation_to_location.iloc[i].BMU_ID,x) for x in predictions],columns=['BMU_ID','predOutput'])])

                #Load precomputed predictions to speed up processing.
                predictions = ((precomputedPred.loc[precomputedPred['BMU_ID'] == allocation_to_location.iloc[i].BMU_ID]).predOutput).array
                for j in range(count):
                    predictionList = np.append(predictionList, (predictions))


        # F1 = mean of all allocations (maxmise)
        #Total predicted output of this allocation as 'Load factor' percentage (% of total capacity allocated)
        meanPredOutput = predictionList.mean() #total / sum(self.genes)
        #F1.5 - Std of predictions (minimise -> * -1)
        deviationOutput = -1 * predictionList.std()
        # F2 = Lower quartile of all locations
        lowerQuartOutput = np.quantile(predictionList, 0.25)
        # F3 = Min?
        minOutput = predictionList.min()
        #Returns Objective (F1,F2)
        # save.to_csv('../../Data/PreComputedPredictions.csv')
        return (meanPredOutput, deviationOutput,lowerQuartOutput)


class Generation:

    def __init__(self,
                 genNumber,
                 size,
                 parameters,
                 fitnessModel,
                 mutationRate=0.1,
                 population=[]):
        self.genNumber = genNumber
        self.popSize = size
        self.geneCount, self.budget = parameters
        self.fitnessModel = fitnessModel
        if population:
            self.population = population
        else:
            self.population = [
                Chromosome(self.geneCount, self.budget)
                for x in range(self.popSize)
            ]

    def getPopulation(self):
        return self.population

    def getPopulationString(self):
        return [c.genes for c in self.population]

    #https://link.springer.com/content/pdf/10.1007%2F978-1-4614-6940-7.pdf - Elitist Non-dominated Sorting GA (NSGA-II)
    def getOffspring(self):
        fitnesses = []
        for chromosome in self.population:
            fitnesses.append(chromosome.fitness(self.fitnessModel))
        # print(fitnesses)

        #Find Non-Dominated Front
        #   Rank By fronts
        #   Rank by crowding distance
    
        nodeDomCount = {}
        
        for fit1 in fitnesses:
            nodeDomCount[fit1] = 0
            for fit2 in fitnesses:
                if (fit1 == fit2):
                    pass
                elif (dominated(fit1, fit2)):
                    if (fit1 in nodeDomCount.keys()):
                        nodeDomCount[fit1] += 1
                    
                    # dominatedNodes.append(fit1)
                    

        plt.figure()
        x = [x[0] for x in fitnesses]
        y = [x[1] for x in fitnesses]
        c = [nodeDomCount[x] for x in fitnesses]
        plt.scatter(x,y,c=c)
        plt.show()

        #Random Tournament Selection
        #



        
        newPop = []
        return Generation(self.genNumber + 1, self.popSize,
                          (self.geneCount, self.budget), newPop)


def dominated(fitness_a, fitness_b):
    if ((fitness_a[0] <= fitness_b[0]) and (fitness_a[1] < fitness_b[1]) or ((fitness_a[0] < fitness_b[0]) and (fitness_a[1] <= fitness_b[1]))):
        return True
    else:
        return False


#Based upon: https://www.sciencedirect.com/science/article/pii/S0020025515007276
def main():
    global allocation_to_location
    # global location_to_expected_weather
    allocation_to_location = pd.read_csv('../../Data//locations.csv').drop(
        'capacity', axis=1)
    predictor = pred.initaliseModel()
    chromosomeParameters = (56, 500)
    gen0 = Generation(0, 100, chromosomeParameters, predictor)
    gen0.getOffspring()


if __name__ == "__main__":
    main()