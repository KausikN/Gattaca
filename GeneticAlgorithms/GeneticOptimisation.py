'''
Optimization using Genetic Operator
'''
# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
# Optimiser Function
def GeneticOpimizer_Basic(equation_inputs, num_weights, sol_per_pop, num_generations, num_parents_mating, FitnessFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc, boundary=(None, None), verbose=False, DisplayWidget=None):
    # The population will have sol_per_pop chromosomes where each chromosome has num_weights genes
    pop_size = (sol_per_pop, num_weights)
    lowerbound = boundary[0]
    upperbound = boundary[1]
    if boundary[0] == None:
        lowerbound = -4.0
    if boundary[1] == None:
        upperbound = 4.0

    # Create the initial population
    new_population = np.random.uniform(low=lowerbound, high=upperbound, size=pop_size)

    History = []
    max_fitness = None
    best_chromosome = None
    for generation in tqdm(range(num_generations)):
        # Measure the fitness of each chromosome in the population.
        fitness = FitnessFunc(equation_inputs, new_population)

        # Print
        if not max_fitness == None and verbose:
            print("Best result after generation", str(generation - 1) + ":", np.max(fitness))
            print("Improvement in result:", str(np.max(fitness) - max_fitness))

        # Update Best Values
        if max_fitness == None or max_fitness < np.max(fitness):
            max_fitness = np.max(fitness)
            best_chromosome = new_population[np.argmax(fitness)]

        # Select the best parents in the population for mating
        parents = ParentSelectorFunc(new_population, fitness, num_parents_mating)

        # Generate next generation using CrossoverFunc
        offspring_crossover = CrossoverFunc(parents, offspring_size=(pop_size[0] - parents.shape[0], num_weights))

        # Add some variations (mutate) to the offsrping using MutationFunc
        offspring_mutation = MutationFunc(offspring_crossover, mutated_gene_index=None, boundary=boundary)
    
        # Save History
        thisGenData = {
            'max_fitness_ingen': np.max(fitness),
            'best_chromosome_ingen': list(new_population[np.argmax(fitness)]),
            'max_fitness': max_fitness,
            'best_chromosome': list(best_chromosome)
        }
        History.append(thisGenData)

        # Prints
        if verbose:
            print("Generation:", str(generation + 1), "\n\n")

            print("Fitness Values:\n")
            print(fitness)
            print("\n")

            print("Selected Parents:\n")
            for p in parents:
                print(p)
            print("\n")

            print("Crossover Result:\n")
            for off in offspring_crossover:
                print(off)
            print("\n")

            print("Mutation Result:\n")
            for off in offspring_mutation:
                print(off)
            print("\n\n")

        # Create the new population based on the parents and offspring
        new_population[0 : parents.shape[0], :] = parents
        new_population[parents.shape[0] : , :] = offspring_mutation

        # If the display widget is given, display the progress
        if DisplayWidget is not None: DisplayWidget.markdown("Gen [" + str(generation+1) + " / " + str(num_generations) + "]: " + "Max Fitness = " + str(max_fitness))

    # If the display widget is given, display the result
    if DisplayWidget is not None: DisplayWidget.markdown("Gen [" + str(num_generations) + " / " + str(num_generations) + "]: " + "Max Fitness = " + str(max_fitness))

    RunData = {
        'equation_inputs': equation_inputs,
        'num_weights': num_weights,
        'sol_per_pop': sol_per_pop,
        'num_generations': num_generations,
        'num_parents_mating': num_parents_mating,
        'FitnessFunc': FitnessFunc,
        'ParentSelectorFunc': ParentSelectorFunc,
        'CrossoverFunc': CrossoverFunc,
        'MutationFunc': MutationFunc,
        'boundary': boundary,

        'max_fitness': max_fitness,
        'best_chromosome': list(best_chromosome),
        'run_history': History,
    }

    return RunData

def GeneticOpimizer_PrintRunSummary(RunData, ncols=1):
    max_fitness = RunData['max_fitness']
    best_chromosome = RunData['best_chromosome']
    equation_inputs = RunData['equation_inputs']
    num_generations = RunData['num_generations']
    max_fitness_ingen_history = [RunData['run_history'][i]['max_fitness_ingen'] for i in range(num_generations)]
    best_chromosome_ingen_history = [RunData['run_history'][i]['best_chromosome_ingen'] for i in range(num_generations)]

    print("Summary:\n")
    # Best Performer Chromosome
    print("Best Fitness:", max_fitness)
    print("Best Chromosome:", best_chromosome)
    print("\n\n")

    # Plots
    # Best Fitness Per Generation Plot
    plt.plot(range(1, num_generations+1), max_fitness_ingen_history)
    plt.show()

    # Best Chromosome Per Generation Plot
    best_chromosome_ingen_history = np.array(best_chromosome_ingen_history)
    n_genes = len(best_chromosome)
    nrows = int(n_genes / ncols) + 1

    gen_range = range(1, num_generations+1)
    for gene_index in range(n_genes):
        ax = plt.subplot(nrows, ncols, gene_index+1)
        ax.title.set_text("Gene " + str(gene_index+1) + ": Input: " + str(equation_inputs[gene_index]) + " , Best: " + str(best_chromosome[gene_index]))
        plt.plot(gen_range, best_chromosome_ingen_history[:, gene_index])
    plt.show()

# Driver Code
# # Params
# verbose = False
# Summary = True

# sol_per_pop = 200
# num_generations = 5000
# num_parents_mating = 100

# ParentSelectorFunc = ParentSelectorFunctions.ParentSelectorFunc_BestFitness
# CrossoverFunc = CrossoverFunctions.CrossoverFunc_MidPoint
# MutationFunc = MutationFunctions.MutationFunc_UniformNoise

# ncols = 1
# # Params

# # RunCode
# # Polynomial Fitting
# # x^3 - 2(x^2) + x  within (0, 31)
# FitnessFunc = FitnessFunctions.FitnessFunc_Polynomial
# boundary = (0, 31)
# equation_inputs = [0, -2, 1]
# num_weights = 1 # Number of the weights we are looking to optimize.
# RunData = GeneticOpimizer_Basic(equation_inputs, num_weights, sol_per_pop, num_generations, num_parents_mating, FitnessFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc, boundary=boundary, verbose=verbose)
# GeneticOpimizer_PrintRunSummary(RunData, ncols=ncols)