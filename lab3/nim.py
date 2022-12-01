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
    cooked["possible_moves"] = [
        (r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1) if state.k is None or o <= state.k
    ]
    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
    cooked["active_rows_index"] = [idx for idx, row in enumerate(state.rows) if row > 0]
    cooked['single_elem_row'] = any([state.rows[i] == 1 for i in cooked['active_rows_index']]) and not all(
        [state.rows[i] == 1 for i in cooked['active_rows_index']])

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


# %% Cell to be able to play game
NUM_MATCHES = 10
NIM_SIZE = 10


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
evaluate(optimal_strategy, optimal_strategy)

# %% A visualized match between optimal strategies

strategy = (optimal_strategy, optimal_strategy)

nim = Nim(4)
logging.debug(f"status: Initial board  -> {nim}")
player = 0
while nim:
    ply = strategy[player](nim)
    nim.nimming(ply)
    logging.debug(f"status: After player {player} -> {nim}")
    player = 1 - player
winner = 1 - player
logging.info(f"status: Player {winner} won!")

# %% Q.2

"""
    Rules to evolve:
        1. if one row --> take elems until x remains
        2. if two rows where one row has one elem --> take elems until x remains
        3. if two rows with multiple elems --> take elems until x remains
"""


# %%

# %% Q.2 Create own strategy based on cooked information
# strategy maker: choose from the shortest row w.p. p and from the longest row w.p 1-p
def make_strategy(genome: dict) -> Callable:
    def evolvable(state: Nim) -> Nimply:
        data = cook_status(state)

        active_rows_num = data['active_rows_number']
        active_rows_index = data['active_rows_index']
        poss_moves = data['possible_moves']

        # logging.debug(f'\nCurrent state: {state} \nNumber of active rows: {active_rows_num}'
        #              f' \nIndex of active rows: {active_rows_index} \nPossible moves: {poss_moves}\n')

        # rule 1
        if active_rows_num == 1:
            #    logging.debug(f'Applying rule 1\n')
            if max([m[1] for m in poss_moves]) <= genome['rules']['rule_1']:
                ply = Nimply(active_rows_index[0], 1)
            else:
                move = [(r, e) for (r, e) in poss_moves if state.rows[r] - e == genome['rules']['rule_1']][0]
                ply = Nimply(move[0], move[1])

        # rule 2
        elif active_rows_num == 2:
            if data['single_elem_row']:
                #        logging.debug(f'Applying rule 2\n')
                if genome['rules']['rule_2'][0] == 0:  # choose from row with single elem
                    row = [i for i, r in enumerate(state.rows) if r == 1][0]
                    elem = 1  # exists only one elem to remove
                    ply = Nimply(row, elem)
                else:  # choose from row with multiple elems
                    row = [i for i in active_rows_index if state.rows[i] != 1][0]
                    elem = max(state.rows[row] - genome['rules']['rule_2'][1], 1)
                    # if genome['rule_2'][1] < state.rows[row]:
                    #    elem = state.rows[row] - genome['rule_2'][1] # remove elem elems until pre-decided elems left
                    # else:
                    #    elem = 1
                    ply = Nimply(row, elem)

            # rule 3
            else:
                #        logging.debug(f'Applying rule 3\n')
                if genome['rules']['rule_3'][0] == 0:  # choose from row with fewest elements
                    row = data['shortest_row']
                else:
                    row = data['longest_row']
                elem = max(state.rows[row] - genome['rules']['rule_3'][1], 1)
                ply = Nimply(row, elem)

        else:
            #    logging.debug(f'Applying rule 4\n')
            move = random.choice(poss_moves)
            ply = Nimply(move[0], move[1])

        # logging.debug(f'Making move {ply}\n')
        return ply

    return evolvable


# make strategy that returns a row and number of elems to remove based on cooked_data
def my_strategy(state: Nim) -> Nimply:
    pass

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


# genome1 = {'rule_1': 2, 'rule_2': }
# evaluate(make_strategy({"p": 0.9999}))
# %% INIT POPULATION

POP_SIZE = 50
OFFSPRING_SIZE = 200
pop = init_population(POP_SIZE, NIM_SIZE)


# %% EVOLUTION STRATEGIES

# compute fitness by head2head-games
def head2head(p1: dict, p2: dict):
    players = (make_strategy(p1), make_strategy(p2))

    nim = Nim(NIM_SIZE)
    player = 0
    while nim:
        ply = players[player](nim)
        nim.nimming(ply)
        player = 1 - player
    winner = 1 - player
    if winner == 0:
        p1['fitness'] += 1
        #logging.info(f'Player {winner} won and has now fitness {p1["fitness"]}')
    else:
        p2['fitness'] += 1
        #logging.info(f'Player {winner} won and has now fitness {p2["fitness"]}')


def calc_fitness(individuals: list) -> None:
    for ind1 in individuals:
        for ind2 in individuals:
            #don't want one individual to play against itself
            if ind1 != ind2:
                head2head(ind1,ind2)


# get k best inds to make offspring from
def k_fittest_individuals(pop: list, k: int) -> list:
    return sorted(pop, key=lambda l: l['fitness'], reverse=True)[:k]


def clean_fitness(individuals: list) -> None:
    for ind in individuals:
        ind['fitness'] = 0


# tournament to decide parents
def tournament(population: list, k: int) -> dict:
    contestors = random.sample(population, k=k)
    clean_fitness(contestors)
    #logging.debug(f'Contestors fitness {[c["fitness"] for c in contestors]} ')
    for p1 in contestors:
        for p2 in contestors:
            if p1!=p2:
                head2head(p1,p2)
    best_contestor = sorted(contestors, key=lambda l: l['fitness'], reverse=True)[0]
    return best_contestor



def cross_over(parent1: dict, parent2: dict, mutation_prob: float) -> dict:
    logging.info(f'Parent 1: {parent1}')
    logging.info(f'Parent 2: {parent2}')
    rules = [rule for rule in parent1['rules'].keys()]
    #rules = [k for k in parent1.keys() if 'rule' in k]
    child = {}
    child['rules'] = {}
    for k in rules:
        which_parent = random.randint(1, 2)
        child['rules'][k] = parent1['rules'][k] if which_parent == 1 else parent2['rules'][k]
    logging.info(f'Child: {child}')
    if random.random() < mutation_prob:
        logging.info(f'Mutating!')
        rule = random.choice(rules)
        r1 = parent1['rules'][rule]
        r2 = parent2['rules'][rule]
        # get average from both parents for rule
        child['rules'][rule] = [int(d) for d in np.mean([r1, r2], axis=0).tolist()]
    logging.info(f'Child: {child}')
    return child

def create_offspring(population: list, k: int, mutation_prob: float) -> list:
    offspring = []
    for i in range(OFFSPRING_SIZE):
        p1 = tournament(population=population, k=k)
        p2 = tournament(population=population, k=k)
        child = cross_over(parent1=p1, parent2=p2, mutation_prob=mutation_prob)
        offspring.append(child)
    return offspring


def get_next_generation(offspring: list) -> list:

#%%
nim_size = 5
for p1 in pop:
    for p2 in pop:
        head2head(p1, p2, nim_size)

# %% play game between two individuals
evaluate(make_strategy(pop[5]), make_strategy(pop[1]))

# %% A visualized match between two individuals
p1 = 5
p2 = 1

strategy = (make_strategy(pop[p1]), make_strategy(pop[p2]))
logging.info(f'Rules Player1: {pop[p1]}')
logging.info(f'Rules Player2: {pop[p2]}')

nim = Nim(4)
logging.debug(f"status: Initial board  -> {nim}")
player = 0
while nim:
    ply = strategy[player](nim)
    nim.nimming(ply)
    logging.debug(f"status: After player {player} -> {nim}")
    player = 1 - player
winner = 1 - player
logging.info(f"status: Player {winner} won!")
