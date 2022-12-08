from typing import Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


class Genetic:

    """
    NOTE:
        - S is the set of members.
        - T is the target value.
        - Chromosomes are represented as an array of 0 and 1 with the same length as the set.
        (0 means the member is not included in the subset, 1 means the member is included in the subset)

        Feel free to add any other function you need.
    """

    def __init__(self):
        pass

    def generate_initial_population(self, n: int, k: int) -> np.ndarray:#todo done
        """
        Generate initial population: This function is used to generate the initial population.

        Inputs:
        - n: number of chromosomes in the population
        - k: number of genes in each chromosome

        It must generate a population of size n for a set of k members.

        Outputs:
        - initial population
        """

        initial_population = []
       # for _ in range(0, n):
       #     initial_population.append(''.join(random.choice(["0", "1"]) for _ in range(k)))
       # return initial_population

        for i in range(n):
            population = []
            for j in range(k):
                population.append(random.randint(0, 1))
            initial_population.append(population)
        return np.array(initial_population)
        pass

    def objective_function(self, chromosome: np.ndarray, S: np.ndarray) -> int:#todo done
        """
        Objective function: This function is used to calculate the sum of the chromosome.

        Inputs:
        - chromosome: chromosome to be evaluated
        - S: set of members

        It must calculate the sum of the members included in the subset (i.e. sum of S[i]s where Chromosome[i] == 1).

        Outputs:
        - sum of the chromosome
        """
       # sum = 0
       # for index, entry in enumerate(S):
       #     if entry == "1":
       #         sum += chromosome[index]
       # return sum#todo: need abs??

        sum = 0
        #for x in range(len(S)):
        #    if chromosome[x] == 1:
        #        sum += S[x]
        ##
        #return sum
        return np.dot(S, chromosome)
        pass

    def is_feasible(self, chromosome: np.ndarray, S: np.ndarray, T: int) -> bool:#todo done
        """
        This function is used to check if the sum of the chromosome (objective function) is equal or less to the target value.

        Inputs:
        - chromosome: chromosome to be evaluated
        - S: set of members
        - T: target value

        Outputs:
        - True (1) if the sum of the chromosome is equal or less to the target value, False (0) otherwise
        """
        if(self.objective_function(chromosome, S) <= T):
            return True
        return False
        pass

    def cost_function(self, chromosome: np.ndarray, S: np.ndarray, T: int) -> int: #todo: done
        """
        Cost function: This function is used to calculate the cost of the chromosome.

        Inputs:
        - chromosome: chromosome to be evaluated
        - S: set of members
        - T: target value

        The cost is calculated in this way:
        - If the chromosome is feasible, the cost is equal to (target value - sum of the chromosome)
        - If the chromosome is not feasible, the cost is equal to the sum of the chromosome

        Outputs:
        - cost of the chromosome
        """
        if(self.is_feasible(chromosome, S, T)):
            cost = T - self.objective_function(chromosome, S)
        else:
            cost = self.objective_function(chromosome, S)

        return cost
        pass

    def selection(self, population: np.ndarray, S: np.ndarray, T: int) -> Tuple[np.ndarray, np.ndarray]:#todo: done but need a check!!!!
        """
        Selection: This function is used to select the best chromosome from the population.

        Inputs:
        - population: current population
        - S: set of members
        - T: target value

        It select the best chromosomes in this way:
        - It gets 4 random chromosomes from the population
        - It calculates the cost of each selected chromosome
        - It selects the chromosome with the lowest cost from the first two selected chromosomes
        - It selects the chromosome with the lowest cost from the last two selected chromosomes
        - It returns the selected chromosomes from two previous steps

        Outputs:
        - two best chromosomes with the lowest cost out of four selected chromosomes
        """

        randomlist = []
        while(True):
            n = random.randint(0, len(population)-1)
            if(n not in randomlist):
                randomlist.append(n)
            if(len(randomlist) == 4):
                break

        cost1 = self.cost_function(population[randomlist[0]], S, T)
        cost2 = self.cost_function(population[randomlist[1]], S, T)
        cost3 = self.cost_function(population[randomlist[2]], S, T)
        cost4 = self.cost_function(population[randomlist[3]], S, T)

        if(cost1 < cost2):
            hold_1 = population[randomlist[0]]
        else:
            hold_1 = population[randomlist[1]]

        if(cost3 < cost4):
            hold_2 = population[randomlist[2]]
        else:
            hold_2 = population[randomlist[3]]

        return (hold_1, hold_2)

        pass

    def crossover(self, parent1: np.ndarray, parent2: np.ndarray, S: np.ndarray, prob: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crossover: This function is used to create two new chromosomes from two parents.

        Inputs:
        - parent1: first parent chromosome
        - parent2: second parent chromosome


        It creates two new chromosomes in this way:
        - It gets a random number between 0 and 1
        - If the random number is less than the crossover probability, it performs the crossover, otherwise it returns the parents
        - Crossover steps:
        -   It gets a random number between 0 and the length of the parents
        -   It creates two new chromosomes by swapping the first part of the first parent with the first part of the second parent and vice versa
        -   It returns the two new chromosomes as children


        Outputs:
        - two children chromosomes
        """

        random_number = random.uniform(0 , 1)
        if(random_number < prob):
            return (parent1, parent2)

        crossover_point = int(random.uniform(0, len(parent1) - 1))
        #new_individual_1 = parent2[:crossover_point] + parent1[crossover_point:]
        #new_individual_2 = parent1[:crossover_point] + parent2[crossover_point:]
        new_individual_1 = np.append(parent2[:crossover_point] , parent1[crossover_point:])
        new_individual_2 = np.append(parent1[:crossover_point] , parent2[crossover_point:])
        return (new_individual_1, new_individual_2)

        pass

    def mutation(self, child1: np.ndarray, child2: np.ndarray, prob: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """
        Mutation: This function is used to mutate the child chromosomes.

        Inputs:
        - child1: first child chromosome
        - child2: second child chromosome
        - prob: mutation probability

        It mutates the child chromosomes in this way:
        - It gets a random number between 0 and 1
        - If the random number is less than the mutation probability, it performs the mutation, otherwise it returns the children
        - Mutation steps:
        -   It gets a random number between 0 and the length of the children
        -   It mutates the first child by swapping the value of the random index of the first child
        -   It mutates the second child by swapping the value of the random index of the second child
        -   It returns the two mutated children

        Outputs:
        - two mutated children chromosomes
        """

        random_number = random.uniform(0 , 1)
        if (random_number < prob):
            return (child1, child2)

        mutation_point = int(random.uniform(0, len(child1) - 1))

        if(child1[mutation_point] == 0):
            child1[mutation_point] = 1
        elif(child1[mutation_point] == 1):
            child1[mutation_point] = 0
        if(child2[mutation_point] == 0):
            child2[mutation_point] = 1
        elif(child2[mutation_point] == 1):
            child2[mutation_point] = 0

        return (child1, child2)

        pass

    def run_algorithm(self, S: np.ndarray, T: int, crossover_probability: float = 0.5, mutation_probability: float = 0.1, population_size: int = 100, num_generations: int = 100):
        """
        Run algorithm: This function is used to run the genetic algorithm.

        Inputs:
        - S: array of integers
        - T: target value

        It runs the genetic algorithm in this way:
        - It generates the initial population todo
        - It iterates for the number of generations todo
        - For each generation, it makes a new empty population
        -   While the size of the new population is less than the initial population size do the following:
        -       It selects the best chromosomes(parents) from the population todo
        -       It performs the crossover on the best chromosomes todo
        -       It performs the mutation on the children chromosomes todo
        -       If the children chromosomes have a lower cost than the parents, add them to the new population, otherwise add the parents to the new population todo what to do??
        -   Update the best cost if the best chromosome in the population has a lower cost than the current best cost
        -   Update the best solution if the best chromosome in the population has a lower cost than the current best solution
        -   Append the current best cost and current best solution to the records list
        -   Update the population with the new population
        - Return the best cost, best solution and records


        Outputs:
        - best cost
        - best solution
        - records
        """

        # UPDATE THESE VARIABLES (best_cost, best_solution, records)
        best_cost = np.Inf
        best_solution = None
        new_cost = 0
        new_state = None
        records = []

        initial_population = self.generate_initial_population(population_size, len(S))

        # YOUR CODE HERE

        for i in tqdm(range(num_generations)):
            new_population = []
            while(len(new_population) < len(initial_population)):
                parents = self.selection(initial_population, S, T)
                cross_parents = self.crossover(parents[0], parents[1], S, crossover_probability)
                mutation_parents = self.mutation(cross_parents[0] , cross_parents[1], mutation_probability)
                mutation_parent0_cost = self.cost_function(mutation_parents[0], S, T)
                mutation_parent1_cost = self.cost_function(mutation_parents[1], S, T)
                parent0_cost = self.cost_function(parents[0], S, T)
                parent1_cost = self.cost_function(parents[1], S, T)
                if(mutation_parent0_cost < parent0_cost):
                    new_population.append(mutation_parents[0])
                    hold1 = mutation_parent0_cost
                    hold_state1 = mutation_parents[0]
                else:
                    new_population.append(parents[0])
                    hold1 = parent0_cost
                    hold_state1 = parents[0]
                if (mutation_parent1_cost < parent1_cost):
                    new_population.append(mutation_parents[1])
                    hold2 = mutation_parent1_cost
                    hold_state2 = mutation_parents[1]
                else:
                    new_population.append(parents[1])
                    hold2 = parent1_cost
                    hold_state2 = parents[1]

                if(hold1 < hold2):
                    new_cost = hold1
                    new_state = hold_state1
                else:
                    new_cost = hold2
                    new_state = hold_state2

            if(new_cost < best_cost):
                best_cost = new_cost
                best_solution = new_state
            initial_population = new_population


            # YOUR CODE HERE

            records.append({'iteration': i, 'best_cost': best_cost,
                           'best_solution': best_solution})  # DO NOT REMOVE THIS LINE

        records = pd.DataFrame(records)  # DO NOT REMOVE THIS LINE

        return best_cost, best_solution, records
