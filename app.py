"""
Stream lit GUI for hosting Gattaca
"""

# Imports
import os
import cv2
import functools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import json

import Gattaca
from GeneticAlgorithms import GeneticOptimisation

from Utils import VideoUtils

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
DEFAULT_PATH_EXAMPLEIMAGE = 'StreamLitGUI/DefaultData/ExampleImage.png'
DEFAULT_SAVEPATH_ANIM = 'StreamLitGUI/DefaultData/SavedAnim.gif'

DEFAULT_IMAGE_SIZE = (256, 256)

# Util Vars
CACHE = {}

# Util Functions
def LoadCache():
    global CACHE
    CACHE = json.load(open(CACHE_PATH, 'r'))

def SaveCache():
    global CACHE
    json.dump(CACHE, open(CACHE_PATH, 'w'), indent=4)

def ResizeImage(I):
    global DEFAULT_IMAGE_SIZE
    aspectRatio = I.shape[0] / I.shape[1]
    ResizeSize = (DEFAULT_IMAGE_SIZE[1], int(DEFAULT_IMAGE_SIZE[1]*aspectRatio))
    I_r = cv2.resize(I, ResizeSize, interpolation=cv2.INTER_NEAREST)
    return I_r

def AddImageElementIndexText(I, i, total, textColor=[255, 255, 255]):
    I_size = I.shape[:2]
    pos = (int(I_size[1]/10), int(I_size[0]/10))

    text = str(i) + "/" + str(total)
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    fontThickness = 1
    padding = [10, 10]

    text_size, _ = cv2.getTextSize(text, font, fontScale, fontThickness)
    text_w, text_h = text_size
    I_t = cv2.rectangle(I, (pos[0] - padding[0], pos[1] + padding[1]), (pos[0] + text_w + padding[0], pos[1] - text_h - padding[1]), [0, 0, 0], -1)
    I_t = cv2.putText(I_t, text, pos, font, fontScale, textColor, fontThickness)
    return I_t

def PlotHeatMap(gridData, title=""):
    fig = plt.Figure()
    ax = fig.add_subplot(111)
    ax = sns.heatmap(gridData, ax=ax)
    plt.title(title)
    # plt.show()
    HeatmapPlot = fig
    return HeatmapPlot

# Main Functions


# UI Functions
def UI_Param(paramData, col=st):
    inp = None
    if paramData['type'] in [float, int]:
        inp = col.number_input(paramData['name'], paramData['min'], paramData['max'], paramData['default'], paramData['step'])
        inp = paramData['type'](inp)
    elif paramData['type'] in [bool]:
        inp = col.checkbox(paramData['name'], paramData['default'])
        inp = paramData['type'](inp)
    
    return inp

def UI_SelectFunc(name, funcChoices, returnAllData=False):
    col1, col2 = st.columns(2)
    USERINPUT_FuncChoice = col1.selectbox("Select " + name + " Function", list(funcChoices.keys()))
    FuncData = funcChoices[USERINPUT_FuncChoice]
    Func = FuncData['func']
    FuncParamsData = FuncData['params']
    FuncParams = {}
    for paramData in FuncParamsData:
        paramInp = UI_Param(paramData, col=col2)
        FuncParams[paramData['name']] = paramInp
    Func = functools.partial(Func, **FuncParams)
    if returnAllData:
        return Func, FuncParams, USERINPUT_FuncChoice
    return Func

def UI_SelectFunctions(ignore_fitness=False):
    st.markdown("## Select Functions")

    # Fitness Function
    FitnessFunc, FitnessEquationStringFunc = None, None
    if not ignore_fitness:
        FitnessFunc, FitnessParams, FitnessName = UI_SelectFunc("Fitness Function", FitnessFunctions.FitnessFuncs, True)
        FitnessEquationStringFunc = functools.partial(FitnessFunctions.FitnessEquationStringFuncs[FitnessName]['func'], **FitnessParams)

    # Parent Selector Function
    ParentSelectorFunc = UI_SelectFunc("Parent Selector Function", ParentSelectorFunctions.ParentSelectorFuncs)

    # Crossover Function
    CrossoverFunc = UI_SelectFunc("Crossover Function", CrossoverFunctions.CrossoverFuncs)

    # Mutation Function
    MutationFunc = UI_SelectFunc("Mutation Function", MutationFunctions.MutationFuncs)

    return FitnessFunc, FitnessEquationStringFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc

def UI_Results(RunData, ncols=3, hugeData=False):
    st.markdown("## Results")

    max_fitness = RunData['max_fitness']
    best_chromosome = RunData['best_chromosome']
    equation_inputs = RunData['equation_inputs']
    num_generations = RunData['num_generations']
    max_fitness_ingen_history = [RunData['run_history'][i]['max_fitness_ingen'] for i in range(num_generations)]
    best_chromosome_ingen_history = [RunData['run_history'][i]['best_chromosome_ingen'] for i in range(num_generations)]
    max_fitness_history = [RunData['run_history'][i]['max_fitness'] for i in range(num_generations)]

    st.markdown("### Overall Results")
    # Equation Inputs
    if not hugeData:
        col1, col2 = st.columns(2)
        col1.markdown("Equation:")
        col2.markdown("```python\n" + str(equation_inputs))

    # Best Fitness Overall
    col1, col2 = st.columns(2)
    col1.markdown("Best Fitness:")
    col2.markdown("```python\n" + str(max_fitness))

    # Best Chromosome Overall
    if not hugeData:
        col1, col2 = st.columns(2)
        col1.markdown("Best Chromosome:")
        col2.markdown("```python\n" + str(best_chromosome))
    else:
        fig = plt.figure()
        plt.title("Best Chromosome")
        plt.plot(range(len(best_chromosome)), best_chromosome)
        st.pyplot(fig)

    st.markdown("### Generation-wise Results")
    fig = plt.figure()
    plt.title("Best Fitness Over Generations")
    plt.plot(range(1, num_generations+1), max_fitness_history)
    st.pyplot(fig)

    fig = plt.figure()
    plt.title("Generation-wise Best Fitness")
    plt.plot(range(1, num_generations+1), max_fitness_ingen_history)
    st.pyplot(fig)

    if hugeData or ncols == 0: return
    st.markdown("### Chromosome Change Results")
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

def UI_LoadImage():
    USERINPUT_RGB = st.checkbox("RGB?")
    USERINPUT_ImageData = st.file_uploader("Upload Start Image", ['png', 'jpg', 'jpeg', 'bmp'])

    if USERINPUT_ImageData is not None:
        USERINPUT_ImageData = USERINPUT_ImageData.read()
    if USERINPUT_ImageData is None:
        USERINPUT_ImageData = open(DEFAULT_PATH_EXAMPLEIMAGE, 'rb').read()

    USERINPUT_Image = cv2.imdecode(np.frombuffer(USERINPUT_ImageData, np.uint8), cv2.IMREAD_COLOR)
    USERINPUT_Image = cv2.cvtColor(USERINPUT_Image, cv2.COLOR_BGR2RGB)

    USERINPUT_ResizeRatio = st.slider("Resize Ratio", 0.0, 1.0, 1.0, 0.01)

    ResizedSize = (int(USERINPUT_Image.shape[1] * USERINPUT_ResizeRatio), int(USERINPUT_Image.shape[0] * USERINPUT_ResizeRatio))
    USERINPUT_Image = cv2.resize(USERINPUT_Image, ResizedSize)

    if not USERINPUT_RGB:
        USERINPUT_Image = cv2.cvtColor(USERINPUT_Image, cv2.COLOR_RGB2GRAY)

    st.image(ResizeImage(USERINPUT_Image), "Input Image " + str(USERINPUT_Image.shape))

    return USERINPUT_Image

def UI_ErodeImage(USERINPUT_Image_Original):
    USERINPUT_ErosionScale = st.slider("Erosion Scale", 0.0, 1.0, 0.1, 0.1)
    Noise = None
    Noise = np.random.uniform(1.0 - USERINPUT_ErosionScale, 1.0 + USERINPUT_ErosionScale, size=USERINPUT_Image_Original.shape)
    USERINPUT_Image_Eroded = np.clip(Noise * USERINPUT_Image_Original, 0, 255)
    USERINPUT_Image_Eroded = np.array(USERINPUT_Image_Eroded, dtype=np.uint8)
    st.image(ResizeImage(USERINPUT_Image_Eroded), caption="Eroded Image")

    return USERINPUT_Image_Eroded, USERINPUT_ErosionScale

# Repo Based Functions
def genetic_equation_optimisation():
    # Title
    st.header("Genetic Equation Optimisation")

    # Prereq Loaders

    # Load Inputs
    st.markdown("## Enter Params")
    USERINPUT_equation_inputs_str = st.text_input("Equation Coeffs (',' separated)", "2, 3")

    col1, col2 = st.columns(2)
    USERINPUT_population_size = col1.number_input("Population Size", 1, 250, 200, 1)
    USERINPUT_num_parents_mating = col2.number_input("Number of Parents Mating", 0, int(USERINPUT_population_size), int(USERINPUT_population_size/2), 1)
    USERINPUT_num_generations = st.number_input("Number of generations", 1, 1000, 200, 1)

    FitnessFunc, FitnessEquationStringFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc = UI_SelectFunctions()

    # Process Inputs
    if USERINPUT_equation_inputs_str == "": return
    equation_inputs = list(map(float, USERINPUT_equation_inputs_str.replace(" ", "").split(',')))
    sol_per_pop = int(USERINPUT_population_size)
    num_weights = int(len(equation_inputs))
    num_parents_mating = int(USERINPUT_num_parents_mating)
    num_generations = int(USERINPUT_num_generations)

    st.markdown("## Optimise Equation")
    st.markdown("```python\n" + FitnessEquationStringFunc(equation_inputs), unsafe_allow_html=True)

    if st.button("Run Optimiser"):
        DisplayWidget = st.empty()
        DisplayWidget.markdown("Started Running...")
        RunData = GeneticOptimisation.GeneticOpimizer_Basic(equation_inputs, num_weights, sol_per_pop, num_generations, num_parents_mating, 
            FitnessFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc, 
            boundary=(None, None), verbose=False, DisplayWidget=DisplayWidget)

        # Display Outputs
        UI_Results(RunData)

def genetic_image_reconstruction():
    # Title
    st.header("Genetic Image Reconstruction")

    # Load Inputs
    USERINPUT_Image_Original = UI_LoadImage()
    USERINPUT_Image_Eroded, ErosionScale = UI_ErodeImage(USERINPUT_Image_Original)

    col1, col2 = st.columns(2)
    USERINPUT_population_size = col1.number_input("Population Size", 1, 250, 200, 1)
    USERINPUT_num_parents_mating = col2.number_input("Number of Parents Mating", 0, int(USERINPUT_population_size), int(USERINPUT_population_size/2), 1)
    USERINPUT_num_generations = st.number_input("Number of generations", 1, 1000, 200, 1)

    _, _, ParentSelectorFunc, CrossoverFunc, MutationFunc = UI_SelectFunctions(ignore_fitness=True)
    MutationFunc = functools.partial(MutationFunc, randomScale=0.1)

    # Process Inputs
    Eroded_Chromosome = GeneticOptimisation.Image2Chromosome(USERINPUT_Image_Eroded)
    Original_Chromosome = GeneticOptimisation.Image2Chromosome(USERINPUT_Image_Original)
    sol_per_pop = int(USERINPUT_population_size)
    num_weights = int(len(list(Eroded_Chromosome)))
    num_parents_mating = int(USERINPUT_num_parents_mating)
    num_generations = int(USERINPUT_num_generations)

    FitnessFunc = functools.partial(FitnessFunctions.FitnessFunc_ReconstructImage, target=Original_Chromosome)

    # Display Outputs
    if st.button("Reconstruct"):
        DisplayWidget = st.empty()
        DisplayWidget.markdown("Started Running...")
        RunData = GeneticOptimisation.GeneticOpimizer_Basic(Eroded_Chromosome, num_weights, sol_per_pop, num_generations, num_parents_mating, 
            FitnessFunc, ParentSelectorFunc, CrossoverFunc, MutationFunc, 
            boundary=(0.0, 1.5), verbose=False, DisplayWidget=DisplayWidget)
        
        # Final Reconstructed Image
        st.markdown("## Reconstructed Image")
        BestChromosome = np.array(RunData['best_chromosome'])
        ReconstructedChromosome = Eroded_Chromosome*BestChromosome
        ReconstructedImage = GeneticOptimisation.Chromosome2Image(ReconstructedChromosome, USERINPUT_Image_Eroded.shape)
        st.image(ResizeImage(ReconstructedImage), caption="Reconstructed Image")

        best_chromosome_grid = np.reshape(RunData['best_chromosome'], USERINPUT_Image_Eroded.shape)
        BestChromosomeHeatMap = PlotHeatMap(best_chromosome_grid, "Best Chromosome HeatMap")
        st.pyplot(BestChromosomeHeatMap)

        # Generation by Generation Best
        # best_chromosome_history = [h['best_chromosome_ingen'] for h in RunData['run_history']]
        best_chromosome_history = [h['best_chromosome'] for h in RunData['run_history']]
        best_image_seq = [GeneticOptimisation.Chromosome2Image(Eroded_Chromosome*np.array(chromosome), USERINPUT_Image_Eroded.shape) for chromosome in best_chromosome_history]
        best_image_seq = [ResizeImage(I) for I in best_image_seq]
        best_image_seq = [AddImageElementIndexText(best_image_seq[i], i+1, len(best_image_seq)) for i in range(len(best_image_seq))]
        VideoUtils.SaveImageSeq(best_image_seq, DEFAULT_SAVEPATH_ANIM)
        # st.video(DEFAULT_SAVEPATH_ANIM)
        st.image(DEFAULT_SAVEPATH_ANIM, caption="Generation Best Animation", use_column_width=True)

        # Results
        UI_Results(RunData, hugeData=True)
    
#############################################################################################################################
# Driver Code
if __name__ == "__main__":
    main()