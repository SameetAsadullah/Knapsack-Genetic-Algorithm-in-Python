import collections
import time
from collections import Counter
import random

import numpy as np
from numpy import concatenate
from numpy import random
from numpy.random import randint

# define variables for input
Item = collections.namedtuple('backpack', 'weight value')  # named tuple to store the weights and values
population_size = 0
max_generations = 0
crossover_probability = 0.0
mutation_probability = 0.0
backpack_capacity = 0
max_weight = 0

# Named Tuple, its values are accessible through fields names as well, e.g Individual.weight e.t.c in this case.
Individual = collections.namedtuple('population', 'cromossome weight value')


# Generate Population , given the size and backback_Capacity
def generate_population(size, backpack_capacity):
    new_population = []

    for i in range(size):
        item = randint(0, backpack_capacity - 1)

        # Initialize Random population
        new_population.append(
            Individual(
                cromossome=randint(2, size=(1, backpack_capacity))[0],
                weight=-1,
                value=-1
            )
        )

    return new_population


# Select a parent from Population
def parent_selection(population):
    parents = []
    total_value = 0

    for individual in population:
        total_value += individual.value

    # Find Fittest Individual to select parent
    highest, second_highest = find_two_fittest_individuals(population)
    parents.append(highest)
    parents.append(second_highest)

    # Check Total sum value of fittest individuals
    sum_value = 0
    while len(parents) < len(population):
        individual = randint(0, len(population) - 1)
        sum_value += population[individual].value

        if sum_value >= total_value:
            parents.append(population[individual])

    return parents


# Apply Crossover on population, Given crossover probability and mutation probabilty
def apply_crossover(population, backpack_capacity, crossover_probability, mutation_probability):
    crossovered_population = []

    while len(crossovered_population) < len(population):
        if randint(0, 100) <= crossover_probability * 100:
            parent_a = randint(0, len(population) - 1)
            parent_b = randint(0, len(population) - 1)

            cromossome_a = concatenate((population[parent_a].cromossome[:int(backpack_capacity / 2)],
                                        population[parent_b].cromossome[int(backpack_capacity / 2):]))
            # Apply Mutation on chromosomes
            cromossome_a = apply_mutation(cromossome_a, backpack_capacity, mutation_probability)

            cromossome_b = concatenate((population[parent_a].cromossome[int(backpack_capacity / 2):],
                                        population[parent_b].cromossome[:int(backpack_capacity / 2)]))
            cromossome_b = apply_mutation(cromossome_b, backpack_capacity, mutation_probability)

            crossovered_population.append(Individual(
                cromossome=cromossome_a,
                weight=-1,
                value=-1
            ))

            crossovered_population.append(Individual(
                cromossome=cromossome_b,
                weight=-1,
                value=-1
            ))

    return crossovered_population


# Calculate Weight value to find fittest indivduals
def _calculate_weight_value(cromossome, backpack):
    weight = 0
    value = 0

    for i, gene in enumerate(cromossome):
        if gene == 1:
            weight += backpack[i].weight
            value += backpack[i].value

    return weight, value


# Apply Mutation on chromosomes , given Mutation probabilty
def apply_mutation(cromossome, backpack_capacity, mutation_probability):
    if randint(0, 100) <= mutation_probability * 100:
        genes = randint(0, 2)

        for i in range(genes):
            gene = randint(0, backpack_capacity - 1)
            if cromossome[gene] == 0:
                cromossome[gene] = 1
            else:
                cromossome[gene] = 0

    return cromossome


# functions to be implemented 

# Find Top 2 Fittest Individual from Population
def find_two_fittest_individuals(population):
    # Your Code Here #
    highest_value = 0
    highest_index = 0
    second_highest_index = 0
    for i in range(len(population)):
        if population[i].value > highest_value:
            highest_value = population[i].value
            highest_index = i;

    highest_value = 0
    for i in range(len(population)):
        if population[i].value > highest_value and i != highest_index:
            highest_val = population[i].value
            second_highest_index = i
    return population[highest_index], population[second_highest_index]


# Calculate Fitness of population, given Items (weight,value) and max weight in action
def calculate_fitness(population, items, max_weight):
    # Your Code Here
    for i in range(len(population)):
        w, v = _calculate_weight_value(population[i].cromossome, items)

        array = population[i].cromossome
        while w > max_weight:
            indices = []
            indices = np.where(array == 1)[0]
            array[random.choice(tuple(indices))] = 0
            w, v = _calculate_weight_value(array, items)

        population[i] = Individual(
            cromossome=array,
            weight=w,
            value=v
        )
    return population


# Run Complete Algorithm Step by step
def runGA():
    population = generate_population(population_size, backpack_capacity)

    print(max_generations)

    value = []
    iteraction = []
    best_solution = None

    for i in range(max_generations):

        # Calculate Fitness of initial population
        fitness = calculate_fitness(population, items, max_weight)

        # Select parent
        parents = parent_selection(fitness)

        # Apply crossover and mutation
        crossovered = apply_crossover(parents, backpack_capacity, crossover_probability, mutation_probability)
        # Calculate Fitness of population
        population = calculate_fitness(crossovered, items, max_weight)
        # Find fittest cadidates
        candidate, _ = find_two_fittest_individuals(population)
        if best_solution is None:
            best_solution = candidate
        elif candidate.value > best_solution.value:
            best_solution = candidate

        value.append(best_solution.value)
        iteraction.append(i)

        # print Every 100th generation results
        if i % 100 == 0:
            print('\nCurrent generation..: {}'.format(i))
            print('Best solution so far: {}'.format(best_solution.value))

    print(' solution found:')
    print('\nWeight: {}'.format(best_solution.weight))
    print('Value.: {}'.format(best_solution.value))
    print('\nBackpack configuration: {}'.format(best_solution.cromossome))


if __name__ == "__main__":
    # Initialize population size and
    # Crossover and Mutation Probabilities

    population_size = randint(50, 200)

    max_generations = randint(100, 1000)
    crossover_probability = round(random.uniform(low=0.3, high=1.0), 1)
    mutation_probability = round(random.uniform(low=0.0, high=0.5), 1)

    # Initialize capacity
    backpack_capacity = randint(10, 20)
    # Max weight
    max_weight = randint(50, 100)

    # Random Initialization of knapsack values
    max_item_weight = 15
    max_item_value = 100

    items = []
    for i in range(backpack_capacity):
        items.append(
            Item(
                weight=randint(1, max_item_weight),
                value=randint(0, max_item_value))
        )

    print('\n\n--- Generated Parameters -----')
    print('Population size......: {}'.format(population_size))
    print('Number of generations: {}'.format(max_generations))
    print('Crossover probability: {}'.format(crossover_probability))
    print('Mutation probability.: {}'.format(mutation_probability))
    print('Backpack capacity....: {}'.format(backpack_capacity))
    print('Max backpack weight..: {}'.format(max_weight))
    runGA()
