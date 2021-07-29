'''
Mutation Functions
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
# Utils Functions


# Mutation Functions
# Mutates the individuals

def MutationFunc_UniformNoise(offspring_crossover, mutated_gene_index=None, boundary=(None, None)):
    # Mutate a single gene in each offspring randomly using uniform noise.

    if mutated_gene_index == None:
        mutated_gene_index = np.random.randint(0, offspring_crossover.shape[1])
    
    for idx in range(offspring_crossover.shape[0]):
        # The random mutation to be added to the gene.
        random_mutation = np.random.uniform(-1.0, 1.0, 1)
        newoff = offspring_crossover[idx, mutated_gene_index] + random_mutation
        if boundary[0] is not None and newoff < boundary[0]:
            continue
        if boundary[1] is not None and newoff > boundary[1]:
            continue
        offspring_crossover[idx, mutated_gene_index] = newoff

    return offspring_crossover

# Main Vars
MutationFuncs = {
    'UniformNoise': {
        "func": MutationFunc_UniformNoise,
        "params": []
    }
}

# Driver Code