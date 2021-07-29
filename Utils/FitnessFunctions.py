'''
Fitness Functions
'''

# Imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Main Functions
# Utils Functions
def Fitness_Plot(FitnessFunc, equation_inputs, start, stop, step):
    x = np.arange(start, stop, step)
    x = np.reshape(x, (x.shape[0], 1))
    fitness = FitnessFunc(equation_inputs, x)
    plt.plot(x, fitness)
    plt.show()

# Fitness Functions
# Calculates the fitness value of each solution in the current population

def FitnessFunc_PolyLinear(equation_inputs, pop):
    # The fitness function calculates the sum of products between each equation input and its corresponding population.
    # eq inputs = [A, B, C, ...]
    # pop = [x1, x2, x3, ...]
    # A(x1) + B(x2) + C(x3) ...

    fitness = np.sum(pop*equation_inputs, axis=1)

    return fitness

def FitnessFunc_Polynomial(equation_inputs, pop):
    # The fitness function calculates the sum of products between each equation input and its corresponding population to an increasing power (Like a polynomial).
    # eq inputs = [A, B, C, ...]
    # pop = [x1, x2, x3, ...]
    # A(x1^0) + B(x2^2) + C(x3^3) ...

    fitness = np.zeros(pop.shape[0])
    for eqi in range(len(equation_inputs)):
        fitVal = np.power(pop[:, eqi], eqi) * equation_inputs[eqi]
        fitness = fitness + fitVal
    
    return fitness

def FitnessFunc_ConvergeTarget(equation_inputs, pop, target=0.0):
    # The fitness function calculates the sum of products between each equation input and its corresponding population and compares to a target value (converging fashion).
    # eq inputs = [A, B, C, ...]
    # pop = [x1, x2, x3, ...]
    # 1 / ((A(x1) + B(x2) + C(x3) ...) - target)

    fitness = 1.0 / np.abs(np.sum(pop*equation_inputs, axis=1) - target)
    
    return fitness

def FitnessFunc_DivergeTarget(equation_inputs, pop, target=0.0):
    # The fitness function calculates the sum of products between each equation input and its corresponding population and compares to a target value (diverging fashion).
    # eq inputs = [A, B, C, ...]
    # pop = [x1, x2, x3, ...]
    # ((A(x1) + B(x2) + C(x3) ...) - target)

    fitness = np.abs(np.sum(pop*equation_inputs, axis=1) - target)
    
    return fitness

# Main Vars
FitnessFuncs = {
    'PolyLinear': {
        "func": FitnessFunc_PolyLinear,
        "params": []
    },
    'Polynomial': {
        "func": FitnessFunc_Polynomial,
        "params": []
    },
    'Converge': {
        "func": FitnessFunc_ConvergeTarget,
        "params": [
            {
                "name": "target",
                "type": float,
                "default": 0.0,
                "min": -1000.0,
                "max": 1000.0,
                "step": 0.5
            }
        ]
    },
    'Diverge': {
        "func": FitnessFunc_DivergeTarget,
        "params": [
            {
                "name": "target",
                "type": float,
                "default": 0.0,
                "min": -1000.0,
                "max": 1000.0,
                "step": 0.5
            }
        ]
    }
}


# Driver Code