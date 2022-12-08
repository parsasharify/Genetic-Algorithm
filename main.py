import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from matplotlib import pyplot as plt
from genetic import Genetic




def read_inputs(path='inputs.pkl'):
    inputs = None
    with open(path, 'rb') as fp:
        inputs = pkl.load(fp)
    return inputs

inputs = read_inputs()



def plot(records):
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost Vs. Iterations")
    plt.plot(records['iteration'], records['best_cost'])
    plt.show()



genetic = Genetic()
for test in inputs:
    bc, bs, records = genetic.run_algorithm(test['S'], test['T'], num_generations=2*len(test['S']))
    print(f"Best Value Found: {genetic.objective_function(bs, test['S'])} - Target Value: {test['T']}")
    plot(records)

