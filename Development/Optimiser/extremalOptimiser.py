from audioop import reverse
from random import random, seed
from re import I
from string import printable
from turtle import color

from sklearn.neural_network import MLPRegressor
import model as pred
import pandas as pd
import numpy as np
import math
from scipy.stats import iqr
from matplotlib import projections, pyplot as plt
import random

global allocation_to_location
# global location_to_expected_weather


class Chromosome:

    def __init__(self, size, budget, genes=[]):
        self.size = size
        self.budget = budget
        #Generate inital random distribution that sums to less than the budget
        if genes:
            self.genes = genes
        else:
            self.genes = (np.random.dirichlet(np.ones(self.size),
                                              size=1)[0]) * self.budget
            self.genes = [math.floor(x) for x in self.genes]

    def getGenes(self):
        return self.genes

    def fitness(self, model):

        #Check doesn't break constraints
        if sum(self.genes) > self.budget:
            return (0, -1, 0, 0)

        #Define Inital values
        predictionList = np.array([])
        # save = pd.DataFrame(data=[],columns=['BMU_ID','predOutput'])
        precomputedPred = pd.read_csv('../../Data/PreComputedPredictions.csv')
        #For each location, model the expected weather, sample that distribution, predict outputs based on that sample
        for i, count in enumerate(self.genes):

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
                #     # smoothedData = (featureData.ewm(alpha=0.5)).mean()
                #     #2 - Take Mean & std of each smoothed weather feature
                #     #3 - Sample 1000 number of data points from normal distribution defined by step 2 (for each feature)
                #     distributions[feature] = np.random.normal(
                #         featureData.mean(), featureData.std(), 25)
                # # 4 - Predict output over those points.
                # predictions = pred.predict(pd.DataFrame(distributions), model)
                # save = pd.concat([save,pd.DataFrame([(allocation_to_location.iloc[i].BMU_ID,x) for x in predictions],columns=['BMU_ID','predOutput'])])

                #Load precomputed predictions to speed up processing.
                predictions = ((precomputedPred.loc[precomputedPred['BMU_ID'] == allocation_to_location.iloc[i].BMU_ID]).predOutput).array
                for j in range(count):
                    predictionList = np.append(predictionList, (predictions))

        # F1 = mean of all allocations (maxmise)
        #Total predicted output of this allocation as 'Load factor' percentage (% of total capacity allocated)
        try:
            meanPredOutput = predictionList.mean()  #total / sum(self.genes)
        except:
            return(0,-1,0,0)
        # F2 = Lower quartile of all locations
        outputIQR = -1 * iqr(predictionList)

        #F3 - Std of predictions (minimise -> * -1)
        deviationOutput = -1 * predictionList.std()

        #Returns Objective (F1,F2)
        # save.to_csv('../../Data/PreComputedPredictions.csv')
        return (meanPredOutput, -1 * np.var(predictionList), deviationOutput,
                outputIQR)


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
        print('Evaluating Initial Pop:\n [',end='')
        for i, chromosome in enumerate(self.population):
            print('.',end='')
            fitnesses.append(chromosome.fitness(self.fitnessModel))
        print(']')
        # print(fitnesses)

        #Find Non-Dominated Front
        #   Rank By fronts
        #   Rank by crowding distance

        dominatedRank = {}
        crowdingDistance = {}

        for i, fit1 in enumerate(fitnesses):
            dominatedRank[i] = 0
            crowdingDistance[i] = 100
            for fit2 in fitnesses:
                if (fit1 != fit2):
                    if (dominated(fit1, fit2)):
                        if (i in dominatedRank.keys()):
                            dominatedRank[i] += 1
                #Calculate Crowding distance
                    crowdingDistance[i] = min(crowdingDistance[i],
                                              distance(fit1, fit2))

        #Random Tournament Selection

        tournamentPool = list(range((self.popSize)))
        selected = []
        while tournamentPool:
            selection1 = tournamentPool.pop(
                random.randint(0,
                               len(tournamentPool) - 1))
            selection2 = tournamentPool.pop(
                random.randint(0,
                               len(tournamentPool) - 1))
            if dominatedRank[selection1] < dominatedRank[selection2]:
                selected.append(selection1)
            elif dominatedRank[selection1] == dominatedRank[selection2]:
                #Compare crowding distance
                if crowdingDistance[selection1] > crowdingDistance[selection2]:
                    selected.append(selection1)
                else:
                    selected.append(selection2)
            else:
                selected.append(selection2)
        #Reproduction
        #Select two random
        #Select Random Crossover point for this generation
        reproductionPool = list(range((self.popSize)))
        offSpring = []
        while reproductionPool:
            chromosome1 = self.population[reproductionPool.pop(
                random.randint(0,
                               len(reproductionPool) - 1))]
            chromosome2 = self.population[reproductionPool.pop(
                random.randint(0,
                               len(reproductionPool) - 1))]
            crossoverPoint = random.randint(1, self.geneCount - 1)

            offSpring.append(
                Chromosome(
                    self.geneCount, self.budget,
                    chromosome1.getGenes()[:crossoverPoint] +
                    chromosome2.getGenes()[crossoverPoint:]))
            offSpring.append(
                Chromosome(
                    self.geneCount, self.budget,
                    chromosome2.getGenes()[:crossoverPoint] +
                    chromosome1.getGenes()[crossoverPoint:]))

        #Mutate a random gene at chance self.mutationRate

        #Combine last gen and offspring into one set, re-sort and take top half as elitist principle
        print('Evaluating Offspring:\n [',end='')
        for i, chromosome in enumerate(offSpring):
            print('.',end='')
            fitnesses.append(chromosome.fitness(self.fitnessModel))
        print(']')
        #Recalculate dominating rank and crowding distance for the combined offspring and original population set.
        for i, fit1 in enumerate(fitnesses):
            dominatedRank[i] = 0
            crowdingDistance[i] = 100
            for fit2 in fitnesses:
                if (fit1 != fit2):
                    if (dominated(fit1, fit2)):
                        if (i in dominatedRank.keys()):
                            dominatedRank[i] += 1
                #Calculate Crowding distance
                    crowdingDistance[i] = min(crowdingDistance[i],
                                              distance(fit1, fit2))


        #Sort by ranking & crowding distance
        print()
        order = (sorted(dominatedRank.items(), key=lambda item: item[1]))
        cutoffPoint = self.popSize
        #If ranking by domination rank doesnt provide a clean cutoff point then return
        if (order[cutoffPoint - 1][1] == order[cutoffPoint][1]):
            splitRankNumber = order[cutoffPoint - 1][1]
            #Add indicies up to the rank that needs to be split
            newPopIndices = [x[0] for x in order if x[1] < splitRankNumber]
            #Get indicies of rank that needs to be sorted
            rankToSort = [x[0] for x in order if x[1] == splitRankNumber]
            splitRankSorted = (sorted(rankToSort,reverse=True,
                                      key=lambda item: crowdingDistance[item]))
            # print(splitRankSorted)
            i = 0
            while len(newPopIndices) < cutoffPoint:
                newPopIndices.append(splitRankSorted[i])
                i += 1
            newPop = [
                self.population[index]
                if index < self.popSize else offSpring[index - self.popSize]
                for index in newPopIndices
            ]

        else:
            #Split at cutoff point no sorting required
            newPopIndices = list(dict(order[:cutoffPoint]).keys())
            #If index refers to an item in original population add that to list, otherwise add from the offspring list
            newPop = [
                self.population[index]
                if index < self.popSize else offSpring[index - self.popSize]
                for index in newPopIndices
            ]

        #Return fitnesses of each resultant generation to plot evolution:
        newGenFitnesses = [fitnesses[i] for i in newPopIndices]
        return Generation(self.genNumber + 1, self.popSize,
                          (self.geneCount, self.budget),self.fitnessModel, population=newPop),newGenFitnesses


def dominated(fitness_a, fitness_b):
    if ((fitness_a[0] <= fitness_b[0]) and (fitness_a[1] < fitness_b[1]) or
        ((fitness_a[0] < fitness_b[0]) and (fitness_a[1] <= fitness_b[1]))):
        return True
    else:
        return False


def distance(fitness_a, fitness_b):
    a = np.array([fitness_a[0], fitness_a[1]])
    b = np.array([fitness_b[0], fitness_b[1]])
    return np.linalg.norm(a - b)


#Based upon: https://www.sciencedirect.com/science/article/pii/S0020025515007276
def main():
    global allocation_to_location
    # global location_to_expected_weather
    allocation_to_location = pd.read_csv('../../Data//locations.csv').drop(
        'capacity', axis=1)
    predictor = pred.initaliseModel()
    max_generations = 25
    chromosomeParameters = (56, 75)
    gen0 = Generation(0, 50, chromosomeParameters, predictor)
    gen_i,genfitness = gen0.getOffspring()
    history = []
    fig,ax = plt.subplots()
    for i in range(0,max_generations):
        print('Generation',i+1,':')
        history.append((i+1,genfitness))
        x_P = [x[0] for x in genfitness]
        y_P = [x[1] for x in genfitness]
        ax.scatter(x_P, y_P, alpha=(1/max_generations)*(i+1),marker='x',color='blue')
        gen_i,genfitness = gen_i.getOffspring()
    #Displat Final Generation:
    x_P = [x[0] for x in genfitness]
    y_P = [x[1] for x in genfitness]
    ax.scatter(x_P, y_P, alpha=(1/max_generations)*(i+1),marker='x',color='green')

    print(gen_i.getPopulationString)

    plt.show()

# y_P = [x[1


if __name__ == "__main__":
    main()