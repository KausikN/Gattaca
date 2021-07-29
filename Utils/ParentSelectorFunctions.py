'''
Parent Selector Functions
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
# Utils Functions


# Parent Selector Functions
# Selects the best individuals in the current generation as parents for producing the offspring of the next generation.

def ParentSelectorFunc_BestFitness(pop, fitness, num_parents):
    # Selects individuals with best fitness

    parents = np.empty((num_parents, pop.shape[1]))
    min_fitness = np.min(fitness)
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = min_fitness - 1
        
    return parents

# Main Vars
ParentSelectorFuncs = {
    'BestFitness': ParentSelectorFunc_BestFitness
}

# Driver Code