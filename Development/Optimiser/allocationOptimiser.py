import model as pred
import pandas as pd
import numpy as np
import math
from scipy.stats import iqr
from matplotlib import pyplot as plt
import seaborn as sns
import sys
global allocation_to_location


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
                predictions = ((precomputedPred.loc[
                    precomputedPred['BMU_ID'] ==
                    allocation_to_location.iloc[i].BMU_ID]).predOutput).array
                for j in range(count):
                    predictionList = np.append(predictionList, (predictions))

        # F1 = mean of all allocations (maxmise)
        #Total predicted output of this allocation as 'Load factor' percentage (% of total capacity allocated)
        try:
            meanPredOutput = predictionList.mean()  #total / sum(self.genes)
        except:
            return (0, -1, 0, 0)

        # F2 = negative variance of allocations (so we can minimise the variance)
        try:
            varianceObj = -1 * np.var(predictionList)
        except:
            return (0, -1, 0, 0)

        #F3 = Maximum output
        try:
            maxOutputObj = predictionList.max()
        except:
            return (0, -1, 0, 0)

        #F4 = Minimum output
        try:
            minOutputObj = predictionList.min()
        except:
            return (0, -1, 0, 0)

        #F5 = Meanoutput * budget allocated
<<<<<<< Updated upstream:Development/Optimiser/allocationOptimiser.py
<<<<<<< Updated upstream:Development/Optimiser/allocationOptimiser.py
        try:
            predOutput = sum(self.genes) * meanPredOutput
        except:
            return (0, -1, 0, 0)

        #Returns Objective (F5,F2,F3,F4)
        # save.to_csv('../../Data/PreComputedPredictions.csv')
        return (meanPredOutput, varianceObj, maxOutputObj, minOutputObj)
=======
        predOutput = sum(self.genes) * meanPredOutput

        #Returns Objective (F5,F2,F3,F4)
        # save.to_csv('../../Data/PreComputedPredictions.csv')
        return (predOutput, varianceObj, maxOutputObj, minOutputObj)
>>>>>>> Stashed changes:Development/Optimiser/extremalOptimiser.py
=======
        predOutput = sum(self.genes) * meanPredOutput

        #Returns Objective (F5,F2,F3,F4)
        # save.to_csv('../../Data/PreComputedPredictions.csv')
        return (predOutput, varianceObj, maxOutputObj, minOutputObj)
>>>>>>> Stashed changes:Development/Optimiser/extremalOptimiser.py


class Generation:

    def __init__(self,
                 genNumber,
                 size,
                 parameters,
                 fitnessModel,
                 mutationRate=0.1,
                 crossoverRate=0.65,
                 population=[]):
        self.genNumber = genNumber
        self.popSize = size
        self.geneCount, self.budget = parameters
        self.fitnessModel = fitnessModel
        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
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
        print('Evaluating Initial Pop:\n [', end='')
        for i, chromosome in enumerate(self.population):
            print('.', end='')
            fitnesses.append(
                (chromosome.fitness(self.fitnessModel), chromosome))
        print(']')

        #Find Non-Dominated Front
        #   Rank By fronts
        #   Rank by crowding distance

        dominatedRank = {}
        crowdingDistance = {}

        for i, (fit1, chromosome) in enumerate(fitnesses):
            dominatedRank[i] = 0
            crowdingDistance[i] = 100
            for (fit2, chromosome) in fitnesses:
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
        while len(selected) < len(tournamentPool):
            selection1 = tournamentPool[np.random.randint(len(tournamentPool))]
            selection2 = tournamentPool[np.random.randint(len(tournamentPool))]
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
        reproductionPool = list(range(len(selected)))
        offSpring = []
        while reproductionPool:
            chromosome1 = self.population[selected[reproductionPool.pop(
                np.random.randint(len(reproductionPool)))]]
            chromosome2 = self.population[selected[reproductionPool.pop(
                np.random.randint(len(reproductionPool)))]]

            #Maybe change to variable 'crossoverRate'
            # crossoverPoint = random.randint(1, self.geneCount - 1)
            crossoverPoint = int(self.crossoverRate * self.geneCount)

            #Mutate a random gene at chance self.mutationRate
            #As encoding does is not binary a bit switch mutation is not applicable.
            #Therefore we will do a random reset of any genes that mutate, as our chromosomes are quite large we probably want to keep mutation rate fairly low
            #Mutating by using the same random generation method as a new chromosome uses
            mutations = (np.random.dirichlet(np.ones(self.geneCount),
                                             size=1)[0]) * self.budget
            mutations = [math.floor(x) for x in mutations]
            offSpringGenes = chromosome1.getGenes(
            )[:crossoverPoint] + chromosome2.getGenes()[crossoverPoint:]
            offSpringGenes = [
                mutations[i] if np.random.uniform() < self.mutationRate else x
                for i, x in enumerate(offSpringGenes)
            ]
            offSpring.append(
                Chromosome(self.geneCount, self.budget, offSpringGenes))

            mutations = (np.random.dirichlet(np.ones(self.geneCount),
                                             size=1)[0]) * self.budget
            mutations = [math.floor(x) for x in mutations]

            offSpringGenes = chromosome2.getGenes(
            )[:crossoverPoint] + chromosome1.getGenes()[crossoverPoint:]
            offSpringGenes = [
                mutations[i] if np.random.uniform() < self.mutationRate else x
                for i, x in enumerate(offSpringGenes)
            ]
            offSpring.append(
                Chromosome(self.geneCount, self.budget, offSpringGenes))

        #Combine last gen and offspring into one set, re-sort and take top half as elitist principle
        print('Evaluating Offspring:\n [', end='')
        for i, chromosome in enumerate(offSpring):
            print('.', end='')
            fitnesses.append(
                (chromosome.fitness(self.fitnessModel), chromosome))
        print(']')
        #Recalculate dominating rank and crowding distance for the combined offspring and original population set.
        for i, (fit1, chromosome) in enumerate(fitnesses):
            dominatedRank[i] = 0
            crowdingDistance[i] = 100
            for (fit2, chromosome) in fitnesses:
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
            splitRankSorted = (sorted(rankToSort,
                                      reverse=False,
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
        topFitnesses = [
            fitness for i, fitness in enumerate(fitnesses)
            if dominatedRank[i] == 0
        ]  #fitnesses[order[0][0]]

        # print(topFitnesses)
        #Return fitnesses of each resultant generation to plot evolution:
        newGenFitnesses = [fitnesses[i][0] for i in newPopIndices]
        return Generation(self.genNumber + 1,
                          self.popSize, (self.geneCount, self.budget),
                          self.fitnessModel,
                          population=newPop), newGenFitnesses, topFitnesses


def dominated(fitness_a, fitness_b):
    #Dominated if worse or equal to all objectives of b & is strictly worse than b in at least one objective
    if (((fitness_a[0] <= fitness_b[0]) and (fitness_a[1] <= fitness_b[1]) and
         (fitness_a[2] <= fitness_b[2]) and (fitness_a[3] <= fitness_b[3])) and
        ((fitness_a[0] < fitness_b[0]) or (fitness_a[1] < fitness_b[1]) or
         (fitness_a[2] < fitness_b[2]) or (fitness_a[3] < fitness_b[3]))):
        return True
    # if ((fitness_a[0] <= fitness_b[0]) and (fitness_a[1] < fitness_b[1]) or
    #     ((fitness_a[0] < fitness_b[0]) and (fitness_a[1] <= fitness_b[1]))):
    #     return True
    else:
        return False


def distance(fitness_a, fitness_b):
    a = np.array([fitness_a[0], fitness_a[1], fitness_a[2],
                  fitness_a[3]])  #,fitness_a[4]])
    b = np.array([fitness_b[0], fitness_b[1], fitness_b[2],
                  fitness_b[3]])  #,fitness_b[4]])
    return np.linalg.norm(a - b)


#Based upon: https://www.sciencedirect.com/science/article/pii/S0020025515007276
def main():
    global allocation_to_location
    allocation_to_location = pd.read_csv('../../Data//locations.csv').drop(
        'capacity', axis=1)
    if len(sys.argv) > 1:
        instance(int(sys.argv[-1]))
    else:
        instance(0)


def instance(seedValue):
    np.random.seed(seedValue)
    predictor = pred.initaliseModel()
    max_generations = 150
    popSize = 100
    chromosomeParameters = (56, 75)
    gen0 = Generation(0, popSize, chromosomeParameters, predictor)
    gen_i, genfitness, topFitness = gen0.getOffspring()
    history = []
    for i in range(0, max_generations):
        history.append((i, genfitness, topFitness))
        print('Generation', i + 1, ':')
        gen_i, genfitness, topFitness = gen_i.getOffspring()
    history.append((i, genfitness, topFitness))

    f = open('AllocationsV2.txt', 'a')
    f.write(
        "\n\nSeed = {} \nPopulation Size = {} \nGenerations = {} \n".format(
            seedValue, popSize, max_generations))
    for fit in topFitness:
        f.write('Fitness = {}\nAllocation:\n{}\n'.format(
            fit[0], fit[1].getGenes()))
    f.close()

    #Plot Each objective against generations
    sns.set()
    fig, axs = plt.subplots(1, 4, constrained_layout=True, figsize=(20, 10))
<<<<<<< Updated upstream:Development/Optimiser/allocationOptimiser.py
<<<<<<< Updated upstream:Development/Optimiser/allocationOptimiser.py
    fig.suptitle('Mean of Top Ranking Allocations') 
    # axs.set
    # axs[0].set_ylabel('Mean Load Factor x Budget Allocated')
    axs[0].set_ylabel('Mean Load Factor')
=======
    fig.suptitle('Mean of Top Ranking Allocations')
    # axs.set
    axs[0].set_ylabel('Mean Load Factor x Budget Allocated')
>>>>>>> Stashed changes:Development/Optimiser/extremalOptimiser.py
=======
    fig.suptitle('Mean of Top Ranking Allocations')
    # axs.set
    axs[0].set_ylabel('Mean Load Factor x Budget Allocated')
>>>>>>> Stashed changes:Development/Optimiser/extremalOptimiser.py
    axs[1].set_ylabel('Variance')
    axs[2].set_ylabel('Max Output')
    axs[3].set_ylabel('Min Output')
    # axs[4].set_ylabel('Budget Allocated')
    #Plot Mean of top fitnesses
    for i in range(4):
        axs[i].set_xlabel('Generations')
        temp = [top for (_, _, top) in history]
        y = []
        for tops in temp:
            sum = 0
            for top in tops:
                sum += top[0][i]
            y.append(sum / len(tops))
        axs[i].plot(y)
    # print(gen_i.getPopulationString())
    plt.savefig('../../Data/ResultsV2/Seed ' + str(seedValue) + '.png')
    plt.show()


if __name__ == "__main__":
    main()
