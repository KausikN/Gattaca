"""
Stream lit GUI for hosting Gattaca
"""

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import json

import Gattaca
from GeneticAlgorithms import GeneticOptimisation

from Utils import FitnessFunctions
from Utils import ParentSelectorFunctions
from Utils import CrossoverFunctions
from Utils import MutationFunctions

# Main Vars
config = json.load(open('./StreamLitGUI/UIConfig.json', 'r'))

# Main Functions
def main():
    # Create Sidebar
    selected_box = st.sidebar.selectbox(
    'Choose one of the following',
        tuple(
            [config['PROJECT_NAME']] + 
            config['PROJECT_MODES']
        )
    )
    
    if selected_box == config['PROJECT_NAME']:
        HomePage()
    else:
        correspondingFuncName = selected_box.replace(' ', '_').lower()
        if correspondingFuncName in globals().keys():
            globals()[correspondingFuncName]()
 

def HomePage():
    st.title(config['PROJECT_NAME'])
    st.markdown('Github Repo: ' + "[" + config['PROJECT_LINK'] + "](" + config['PROJECT_LINK'] + ")")
    st.markdown(config['PROJECT_DESC'])

    # st.write(open(config['PROJECT_README'], 'r').read())

#############################################################################################################################
# Repo Based Vars
CACHE_PATH = "StreamLitGUI/CacheData/Cache.json"

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(CACHE_PATH, 'r'))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(CACHE_PATH, 'w'), indent=4)

# Main Functions


# UI Functions
def UI_SelectFunctions():
    st.markdown("## Select Functions")

    # Fitness Function
    USERINPUT_FitnessChoice = st.selectbox("Select Fitness Function", list(FitnessFunctions.FitnessFuncs.keys()))
    FitnessFunc = FitnessFunctions.FitnessFuncs[USERINPUT_FitnessChoice]

    # Parent Selector Function
    USERINPUT_ParentSelectorChoice = st.selectbox("Select Parent Selector Function", list(ParentSelectorFunctions.ParentSelectorFuncs.keys()))
    ParentSelectorFunc = ParentSelectorFunctions.ParentSelectorFuncs[USERINPUT_ParentSelectorChoice]

    # Crossover Function
    USERINPUT_CrossoverChoice = st.selectbox("Select Crossover Function", list(CrossoverFunctions.CrossoverFuncs.keys()))
    CrossoverFunc = CrossoverFunctions.CrossoverFuncs[USERINPUT_CrossoverChoice]

    # Mutation Function
    USERINPUT_MutationChoice = st.selectbox("Select Mutation Function", list(MutationFunctions.MutationFuncs.keys()))
    MutationFunc = MutationFunctions.MutationFuncs[USERINPUT_MutationChoice]

    return FitnessFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc

def UI_Results(RunData, ncols=3):
    max_fitness = RunData['max_fitness']
    best_chromosome = RunData['best_chromosome']
    equation_inputs = RunData['equation_inputs']
    num_generations = RunData['num_generations']
    max_fitness_ingen_history = [RunData['run_history'][i]['max_fitness_ingen'] for i in range(num_generations)]
    best_chromosome_ingen_history = [RunData['run_history'][i]['best_chromosome_ingen'] for i in range(num_generations)]

    col1, col2 = st.beta_columns(2)
    col1.markdown("Equation:")
    col2.markdown("```python\n" + str(equation_inputs))

    col1, col2 = st.beta_columns(2)
    col1.markdown("Best Chromosome:")
    col2.markdown("```python\n" + str(best_chromosome))
    
    col1, col2 = st.beta_columns(2)
    col1.markdown("Best Fitness:")
    col2.markdown("```python\n" + str(max_fitness))

    fig = plt.figure()
    plt.title("Best Fitness Over Generations")
    plt.plot(range(1, num_generations+1), max_fitness_ingen_history)
    st.pyplot(fig)

    fig = plt.figure()
    best_chromosome_ingen_history = np.array(best_chromosome_ingen_history)
    n_genes = len(best_chromosome)
    nrows = int(n_genes / ncols) + 1
    gen_range = range(1, num_generations+1)
    for gene_index in range(n_genes):
        ax = plt.subplot(nrows, ncols, gene_index+1)
        ax.title.set_text("Gene " + str(gene_index+1) + ": Input: " + str(equation_inputs[gene_index]) + " , Best: " + str(best_chromosome[gene_index]))
        plt.plot(gen_range, best_chromosome_ingen_history[:, gene_index])
    st.pyplot(fig)



# Repo Based Functions
def genetic_equation_optimisation():
    # Title
    st.header("Genetic Equation Optimisation")

    # Prereq Loaders

    # Load Inputs
    st.markdown("## Enter Params")
    USERINPUT_equation_inputs_str = st.text_input("Equation Coeffs (',' separated)")
    col1, col2 = st.beta_columns(2)
    USERINPUT_population_size = col1.number_input("Population Size", 1, 250, 200, 1)
    USERINPUT_num_parents_mating = col2.number_input("Number of Parents Mating", 0, int(USERINPUT_population_size), int(USERINPUT_population_size/2), 1)
    USERINPUT_num_generations = st.number_input("Number of generations", 1, 1000, 200, 1)

    FitnessFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc = UI_SelectFunctions()

    # Process Inputs
    if USERINPUT_equation_inputs_str == "": return
    equation_inputs = list(map(float, USERINPUT_equation_inputs_str.replace(" ", "").split(',')))
    sol_per_pop = int(USERINPUT_population_size)
    num_weights = int(len(equation_inputs))
    num_parents_mating = int(USERINPUT_num_parents_mating)
    num_generations = int(USERINPUT_num_generations)
    if st.button("Run Optimiser"):
        DisplayWidget = st.empty()
        DisplayWidget.markdown("Started Running...")
        RunData = GeneticOptimisation.GeneticOpimizer_Basic(equation_inputs, num_weights, sol_per_pop, num_generations, num_parents_mating, 
            FitnessFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc, 
            boundary=(None, None), verbose=False, DisplayWidget=DisplayWidget)

        # Display Outputs
        st.markdown("## Results")
        UI_Results(RunData)

    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()