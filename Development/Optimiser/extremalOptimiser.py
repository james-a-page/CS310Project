import model as pred
import pandas as pd
import numpy as np
import math

global allocation_to_location
global location_to_expected_weather

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
    for i,count in enumerate(allocation):
        print(allocation_to_location.iloc[i].BMU_ID)
        # pred.predict(,model)
    
    return 0

#Based upon: https://www.sciencedirect.com/science/article/pii/S0020025515007276
def main():
    global allocation_to_location
    global location_to_expected_weather
    allocation_to_location = pd.read_csv('../../Data//locations.csv').drop('capacity',axis=1)
    predictor = pred.initaliseModel()
    chromosomeParameters = (5,100)
    gen0 = Generation(0,10,chromosomeParameters,predictor)
    print(gen0.getPopulation())
    fitness(gen0.getPopulation()[0],predictor)




if __name__ == "__main__":
    main()