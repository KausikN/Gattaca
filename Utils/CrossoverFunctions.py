'''
Crossover Functions
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
# Utils Functions


# Crossover Functions
# Crossover the parents

def CrossoverFunc_MidPoint(parents, offspring_size):
    # Crossovers the parents at the center

    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        
    return offspring

# Main Vars
CrossoverFuncs = {
    'MidPoint': CrossoverFunc_MidPoint
}

# Driver Code