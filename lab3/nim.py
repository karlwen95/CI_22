"""
Agents based on different strategies playing Nim (description here: https://en.wikipedia.org/wiki/Nim)
    1. Agent based on rules
    2. Agent based on evolved rules
    3. Agent using minmax
    4. Agent using reinforcement learning

@Author: Karl Wennerström in collaboration with Erik Bengtsson (s306792)
"""

# %% IMPORTS
import logging
from collections import namedtuple
import random
from typing import Callable
from copy import deepcopy
from itertools import accumulate
from operator import xor
import numpy as np
from tqdm import tqdm
from evolvable_agent import Evolvable_agent

# %% SET LOGGING CONFIGS
logging.getLogger().setLevel(logging.DEBUG)

# %% Nim class (created by Prof. Giovanni Squillero)
Nimply = namedtuple("Nimply", "row, num_objects")


class Nim:
    def __init__(self, num_rows: int, k: int = None) -> None:
        self._rows = [i * 2 + 1 for i in range(num_rows)]
        self._k = k

    def __bool__(self):
        return sum(self._rows) > 0

    def __str__(self):
        return "<" + " ".join(str(_) for _ in self._rows) + ">"

    @property
    def rows(self) -> tuple:
        return tuple(self._rows)

    @property
    def k(self) -> int:
        return self._k

    def nimming(self, ply: Nimply) -> None:
        row, num_objects = ply
        assert self._rows[row] >= num_objects
        assert self._k is None or num_objects <= self._k
        self._rows[row] -= num_objects


# %% Extract valuable information, TODO: update to get reasonable rules to evolve in Q.2
def cook_status(state: Nim) -> dict:
    cooked = dict()
    cooked["rows"] = state.rows
    cooked["possible_moves"] = [
        (r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1) if state.k is None or o <= state.k
    ]
    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
    cooked["active_rows_index"] = [idx for idx, row in enumerate(state.rows) if row > 0]
    cooked['single_elem_row'] = any([state.rows[i] == 1 for i in cooked['active_rows_index']]) and not all(
        [state.rows[i] == 1 for i in cooked['active_rows_index']])
    cooked['single_elem_rows_index'] = [i for i, r in enumerate(state.rows) if r == 1]

    cooked["shortest_row"] = min((x for x in enumerate(state.rows) if x[1] > 0), key=lambda y: y[1])[0]
    cooked["longest_row"] = max((x for x in enumerate(state.rows)), key=lambda y: y[1])[0]
    cooked["nim_sum"] = nim_sum(state)

    brute_force = list()
    for m in cooked["possible_moves"]:
        tmp = deepcopy(state)
        tmp.nimming(m)
        brute_force.append((m, nim_sum(tmp)))
    cooked["brute_force"] = brute_force

    return cooked


# %% Q.1
"""
    Use the nim_sum and optimal_strategy from Prof. Squillero.
    Play the optimal_strategy vs itself 
"""


# %% Nim-sum strategy
def nim_sum(state: Nim) -> int:
    *_, result = accumulate(state.rows, xor)
    return result


# optimal strategy making a move generating nim_sum == 0
def optimal_strategy(state: Nim) -> Nimply:
    data = cook_status(state)
    return next((bf for bf in data["brute_force"] if bf[1] == 0), random.choice(data["brute_force"]))[0]

#%% Evaluate the nim-sum strategy
def evaluate(strategy1: Callable, strategy2: Callable) -> float:
    players = (strategy1, strategy2)
    won = 0

    for m in range(NUM_MATCHES):
        nim = Nim(NIM_SIZE)
        player = 0
        while nim:
            ply = players[player](nim)
            nim.nimming(ply)
            player = 1 - player
        if player == 1:
            won += 1
    return won / NUM_MATCHES

starting_wins = evaluate(optimal_strategy, optimal_strategy)
print(f'Optimal strategy wins {starting_wins*100: .0f}% when starting and {(1-starting_wins)*100: .0f}% when not starting.')


# %% Q.2

"""
    Rules to evolve:
        1. if one row --> take elems until x remains
        2. if two rows where one row has one elem --> take elems until x remains
        3. if two rows with multiple elems --> take elems until x remains
        TODO: find rules for how to play when > 2 active rows (except random)
        4*. agent's + oponent's move should be certain sum?
        
    Regarding rules from prof. Squillero:
        * Can evolve order
        * Parameters to tune, like a weight of importance to follow a rule/make a move
        * have agents that ranges form REALLY BAD --> Expert (nim-sum) [Dumb, random, nim-sum]
        * Impressive if algorithm beats human 
        * fitness can be winning percentage against different agents (as a tuple)
"""
# %% Q.2 Create own strategy based on cooked information

# strategy maker: play by the rules
def make_strategy(agent: Evolvable_agent) -> Callable:
    def evolvable(state: Nim) -> Nimply:
        data = cook_status(state)

        # rule 1
        if data['active_rows_number'] == 1:
            row, elem = agent.rule1(data)
            ply = Nimply(row,elem)

        # rule 2
        elif data['active_rows_number'] == 2:
            if data['single_elem_row']:
                row, elem = agent.rule2(data)
                ply = Nimply(row,elem)

            # rule 3
            else:
                row, elem = agent.rule3(data)
                ply = Nimply(row, elem)

        else:
            row, elem = agent.rule4(data)
            ply = Nimply(row, elem)

        return ply

    return evolvable


# human strategy, make moves through input
def my_strategy(state: Nim) -> Nimply:
    print(f'Current state: {state.rows}')
    data = cook_status(state)
    pm = data['possible_moves']
    index = input(f'Choose a play: {[(i,m) for i,m in enumerate(pm)]}')
    while True:
        try:
            assert int(index) in range(len(pm))
        except Exception:
            print('Invalid input, try again')
            index = input(f'Choose a play: {[(i, m) for i, m in enumerate(pm)]}')
        else:
            row = pm[int(index)][0]
            elems = pm[int(index)][1]
            break
    return Nimply(row, elems)


# dumb strategy (to evaluate my agent)
def dumb_agent(state: Nim) -> Nimply:
    """
    Make stupid move. Always remove 1 from shortest row
    """
    data = cook_status(state)
    row = data['shortest_row']
    return Nimply(row, 1)


# random strategy (to evaluate my agent)
def pure_random(state: Nim) -> Nimply:
    """Agent playing completely random"""
    row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
    num_objects = random.randint(1, state.rows[row])
    return Nimply(row, num_objects)


#%% EVOLUTION STRATEGY DESCRIBED

"""
(mu, lambda)-strategy
    1. Create population with the same set of rules but different parameters for each rule
    2. k individuals competes in a tournament where the winner becomes a parent
    3. Perform cross_over between two parents and mutate (aggregate random rule, e.g. mean(both parents' rule)) with certain prob
    4. Generate offspring where OFFSPRING_SIZE>>POPULATION_SIZE
    5. Fitness for offsprings corresponds to how many games are won against their 'siblings' 
    6. Slice new population from fittest offpring
    7. Repeat step 2-6 GENERATION times 
"""



# %% create population with different rule-settings

# currently: 3 rules --> (nim_size-1)*2+2*(nim_size-1)*2+2*(nim_size-1)*2
def init_population(pop_size: int, nim_size: int):
    pop = []
    max_leaveInRow = (nim_size - 1) * 2  # if 19 largest row should at most be able to leave 18
    for i in range(pop_size):
        ind = {}
        ind['rules'] = {}
        # setting for rule 1
        ind['rules']['rule_1'] = random.randint(0, max_leaveInRow)

        # setting for rule 2
        ind['rules']['rule_2'] = [random.randint(0, 1), random.randint(0, max_leaveInRow)]

        # setting for rule 3
        ind['rules']['rule_3'] = [random.randint(0, 1), random.randint(0, max_leaveInRow)]

        ind['fitness'] = 0
        # add ind to pop
        pop.append(ind)
    return pop

# %% EVOLUTION STRATEGIES

def calc_fitness(individuals: list) -> None:
    for ind in individuals:
        fitness = []
        for idx, strat in enumerate(OPPONENTS):
            wins = 0
            for match in range(NUM_MATCHES):
                #don't want one individual to play against itself
                #if ind1 != ind2:
                wins += head2head(ind, strat)
            fitness.append(wins)
        ind['fitness'] = tuple(fitness)


# compute fitness by head2head-games
def head2head(ind: dict, opponent: Callable):
    players = (make_strategy(ind), opponent)

    nim = Nim(NIM_SIZE)
    player = 0
    while nim:
        ply = players[player](nim)
        nim.nimming(ply)
        player = 1 - player
    winner = 1 - player
    if winner == 0:
        return 1
    else:
        return 0


# get k best inds to make offspring from
def k_fittest_individuals(pop: list, k: int) -> list:
    return sorted(pop, key=lambda l: l['fitness'], reverse=True)[:k]


# if want to reset fitness
def clean_fitness(individuals: list) -> None:
    for ind in individuals:
        ind['fitness'] = 0


# tournament to decide parents
def tournament(population: list, k: int) -> dict:
    contestors = random.sample(population, k=k)
    best_contestor = sorted(contestors, key=lambda l: l['fitness'], reverse=True)[0]
    return best_contestor


def cross_over(parent1: dict, parent2: dict, mutation_prob: float) -> dict:
    rules = [rule for rule in parent1['rules'].keys()]
    child = {}
    child['rules'] = {}
    for k in rules:
        which_parent = random.randint(1, 2)
        child['rules'][k] = parent1['rules'][k] if which_parent == 1 else parent2['rules'][k]
    if random.random() < mutation_prob:
        rule = random.choice(rules)
        r1 = parent1['rules'][rule]
        r2 = parent2['rules'][rule]
        child['rules'][rule] = [int(d) for d in np.mean([r1, r2], axis=0).tolist()] if type(r1) == list else int(np.mean([r1, r2]))
    return child


def create_offspring(population: list, k: int, mutation_prob: float) -> list:
    offspring = []
    for _ in range(OFFSPRING_SIZE):
        p1 = tournament(population=population, k=k)
        p2 = tournament(population=population, k=k)
        child = cross_over(parent1=p1, parent2=p2, mutation_prob=mutation_prob)
        offspring.append(child)
    return offspring


def get_next_generation(offspring: list) -> list:
    calc_fitness(offspring)
    return k_fittest_individuals(offspring, POP_SIZE)


#%% MAIN
import sys

if '__name__' == '__main__':
    q = sys.argv[1]
    if q == 1:
        # set params
        NIM_SIZE = 5
        NUM_MATCHES = 10

        # play the nim-sum strategy
        starting_wins = evaluate(optimal_strategy, optimal_strategy)
        print(f'Optimal strategy wins {starting_wins * 100: .0f}% when starting and {(1 - starting_wins) * 100: .0f}% when not starting.')

    elif q == 2:
        # set params
        NIM_SIZE = 5
        NUM_MATCHES = 10
        POP_SIZE = 50
        OFFSPRING_SIZE = 200
        GENERATIONS = 10
        OPPONENTS = [dumb_agent, pure_random, optimal_strategy]

        pop = init_population(POP_SIZE, NIM_SIZE)

    tournament_size = 10
    mutation_prob = 0.3

    for gen in tqdm(range(GENERATIONS), desc='Generations'):
        calc_fitness(pop)
        offspring = create_offspring(pop, tournament_size, mutation_prob)
        pop = get_next_generation(offspring)


# %% A visualized match between two individuals

def play_nim(strategy1, strategy2):
    strategy = (strategy1, strategy2)
    nim = Nim(NIM_SIZE)
    logging.debug(f"status: Initial board  -> {nim}")
    player = 0
    while nim:
        ply = strategy[player](nim)
        nim.nimming(ply)
        logging.debug(f"status: After player {player} -> {nim}")
        player = 1 - player
    winner = 1 - player
    logging.info(f"status: Player {winner} won!")


#%% PLAY

#play_nim(make_strategy(pop[0]), my_strategy)





#%% REGARDING POLICY AND RL

"""
    RL:
        * reward not certain to be instant

"""


#%% UNUSED THINGS

# %% Cell to be able to play game
def evaluate(strategy1: Callable, strategy2: Callable) -> float:
    players = (strategy1, strategy2)
    won = 0

    for m in range(NUM_MATCHES):
        nim = Nim(NIM_SIZE)
        player = 0
        while nim:
            ply = players[player](nim)
            nim.nimming(ply)
            player = 1 - player
        if player == 1:
            won += 1
    return won / NUM_MATCHES

# %% PLAYING: optimal strategy vs optimal strategy
#evaluate(make_strategy(pop[0]), optimal_strategy)


# %% A visualized match between optimal strategies

# strategy = (optimal_strategy, optimal_strategy)
#
# nim = Nim(4)
# logging.debug(f"status: Initial board  -> {nim}")
# player = 0
# while nim:
#     ply = strategy[player](nim)
#     nim.nimming(ply)
#     logging.debug(f"status: After player {player} -> {nim}")
#     player = 1 - player
# winner = 1 - player
# logging.info(f"status: Player {winner} won!")


# %% play game between two individuals
#evaluate(make_strategy(pop[5]), make_strategy(pop[1]))