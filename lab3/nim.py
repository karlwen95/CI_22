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

# TODO: make semi-human smart opponent (with rule 1-3 for example)

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


# %% Extract valuable information
def cook_status(state: Nim) -> dict:
    cooked = dict()
    cooked["rows"] = state.rows
    cooked["possible_moves"] = [
        (r, o) for r, c in enumerate(state.rows) for o in range(1, c + 1) if state.k is None or o <= state.k
    ]
    cooked["active_rows_number"] = sum(o > 0 for o in state.rows)
    cooked["active_rows_index"] = [idx for idx, row in enumerate(state.rows) if row > 0]
    cooked['one_single_elem_row'] = sum([state.rows[i] == 1 for i in cooked['active_rows_index']]) == 1
    cooked['one_multiple_elem_row'] = sum([state.rows[i] > 1 for i in cooked['active_rows_index']]) == 1

    # any([state.rows[i] == 1 for i in cooked['active_rows_index']]) and not all(
    # [state.rows[i] == 1 for i in cooked['active_rows_index']])
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


# %% Q.2

"""
    Rules to evolve:
        1. if one row --> take elems until x remains (optimal: remove all)
        2. if even rows where one row has one elem --> 
            take elems until x remains (optimal: remove all but one from row with mutiple elems)
        3. if odd rows where one row has one elem --> 
            take elems until x remains (optimal: remove all from row with multiple elems)
        4. if even rows with multiple elems --> take elems from longest or shortest row until x remains 
            (what is optimal? try to get situations for rule 1-3?)
        5. if odd rows with multiple elems --> take elems from longest or shortest row until x remains 
            (what is optimal? try to get situations for rule 1-3?)
        6. Play at random (will never get to this rule though...)
        
        # TODO: update rules 4 and 5 to "help" agent get to a situation where rule 1-3 can be used 
            (since these are easy to optmize)
        
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
            ply = Nimply(row, elem)

        elif data['one_multiple_elem_row']:  # all rows but one have a single elem

            # rule2
            if data['active_rows_number'] % 2 == 0:  # even rows
                row, elem = agent.rule2(data)
                ply = Nimply(row, elem)

            # rule 3
            else:  # odd rows
                row, elem = agent.rule3(data)
                ply = Nimply(row, elem)

        elif not data['one_multiple_elem_row']:  # multiple rows are with multiple elems (or also only ones)

            # rule 4
            if data['active_rows_number'] % 2 == 0:
                row, elem = agent.rule4(data)
                ply = Nimply(row, elem)

            # rule 5
            else:
                row, elem = agent.rule5(data)
                ply = Nimply(row, elem)


        else:
            # rule 6 (will we ever get here?)
            logging.info(f'RULE 6!!! Board = {state.rows}')
            row, elem = agent.rule6(data)
            ply = Nimply(row, elem)

        return ply

    return evolvable


# human strategy, make moves through input
def my_strategy(state: Nim) -> Nimply:
    print(f'Current state: {state.rows}')
    data = cook_status(state)
    pm = data['possible_moves']
    index = input(f'Choose a play: {[(i, m) for i, m in enumerate(pm)]}')
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


def semi_smart(state: Nim) -> Nimply:
    """ Make use of rule 1-3, else random"""
    data = cook_status(state)

    if data['active_rows_number'] == 1:
        row = data['active_rows_index'][0]
        elems = state.rows[row]
        ply = Nimply(row, elems)

    elif data['one_multiple_elem_row']:  # all rows but one have a single elem
        if data['active_rows_number'] % 2 == 0:
            move = [(r, e) for (r, e) in data["possible_moves"] if state.rows[r] - e == 1][0]
            ply = Nimply(move[0], move[1])
        else:
            move = [(r, e) for (r, e) in data["possible_moves"] if
                    state.rows[r] - e == 0 and r not in data['single_elem_rows_index']][0]
            ply = Nimply(move[0], move[1])
    else:
        row = random.choice([r for r, c in enumerate(state.rows) if c > 0])
        num_objects = random.randint(1, state.rows[row])
        ply = Nimply(row, num_objects)
    return ply


# %% EVOLUTION STRATEGY DESCRIBED

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


# %% Evolution strategy-functions
def init_population():
    """Initialize population"""
    pop = []
    for i in range(POPULATION_SIZE):
        pop.append(Evolvable_agent(NIM_SIZE))
    return pop


def calc_fitness(individuals: list) -> None:
    """Calculate fitness for each individual as a proportion of won games against different opponents"""
    for ind in individuals:
        fitness = []
        for idx, strat in enumerate(OPPONENTS):
            wins = 0
            for match in range(NUM_MATCHES):
                wins += head2head(ind, strat)
            fitness.append(wins / NUM_MATCHES)
        ind.fitness = tuple(fitness)


# compute fitness by head2head-games
def head2head(agent: Evolvable_agent, opponent: Callable):
    """One game between evolvable agent and opponent"""
    players = (make_strategy(agent), opponent)

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


def fittest_individuals(pop: list) -> list:
    """Return the most fit individuals to use in offspring generation"""
    return sorted(pop, key=lambda l: l.fitness, reverse=True)[:POPULATION_SIZE]


# tournament to decide parents
def tournament(population: list, k: int) -> dict:
    """Select best individual out of k competing in a tournament"""
    contestors = random.sample(population, k=k)
    best_contestor = sorted(contestors, key=lambda l: l.fitness, reverse=True)[0]
    return best_contestor


def cross_over(parent1: Evolvable_agent, parent2: Evolvable_agent, mutation_prob: float) -> Evolvable_agent:
    """Generate new individual by cross-over of parents' rules"""
    rules = [rule for rule in parent1.rules.keys()]
    new_rules = {}
    child = Evolvable_agent(NIM_SIZE)
    for k in rules:
        which_parent = random.randint(1, 2)
        new_rules[k] = parent1.rules[k] if which_parent == 1 else parent2.rules[k]
    if random.random() < mutation_prob:
        rule = random.choice(rules)
        if rule == 'rule_1':
            new_rules[rule] = random.randint(0, (NIM_SIZE - 1) * 2)
        else:
            new_rules[rule] = [random.randint(0, 1), random.randint(0, (NIM_SIZE - 1) * 2)]
    child.rules = new_rules
    return child


def create_offspring(population: list, k: int, mutation_prob: float) -> list:
    """Create new offspring"""
    offspring = []
    for _ in range(OFFSPRING_SIZE):
        p1 = tournament(population=population, k=k)
        p2 = tournament(population=population, k=k)
        child = cross_over(parent1=p1, parent2=p2, mutation_prob=mutation_prob)
        offspring.append(child)
    return offspring


def get_next_generation(offspring: list) -> list:
    """Find the best individuals in the new generation"""
    calc_fitness(offspring)
    return fittest_individuals(offspring)


# %% PLAYING FUNCTIONS
def evaluate(strategy1: Callable, strategy2: Callable) -> float:
    """Play two strategies against each other and evaluate their performance """
    players = (strategy1, strategy2)
    won = 0

    for m in range(EVAL_MATCHES):
        nim = Nim(NIM_SIZE)
        player = 0
        while nim:
            ply = players[player](nim)
            nim.nimming(ply)
            player = 1 - player
        if player == 1:
            won += 1
    print(f'{strategy1.__name__} wins {won*100/EVAL_MATCHES} % of the games against {strategy2.__name__}')
    return won / EVAL_MATCHES


def play_nim(strategy1, strategy2):
    """A visualized match between two strategies"""
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


# %% Q3 - MINMAX AGENT

"""
    Build a minmax agent that alwasy minimizes the opponents maximum win
    Play against optimal strategy, should be able to win if start
    Build as class or function?
    Need:
        keep value for each state (exhaustive)
        condition: return 1 if win -1 else
        condition: return 0 if not decided
            play intil determined and traverse back to that state
"""


# %% MINMAX fcn


def minmax(state: Nim, my_turn: bool, alpha=-1, beta=1):
    if not state:  # empty board then I lose
        return -1 if my_turn else 1

    data = cook_status(state)
    possible_new_states = []
    for ply in data['possible_moves']:
        tmp_state = deepcopy(state)
        tmp_state.nimming(ply)
        possible_new_states.append(tmp_state)
    if my_turn:
        bestVal = -np.inf
        for new_state in possible_new_states:
            value = minmax(new_state, False, alpha, beta)
            bestVal = max(bestVal, value)
            alpha = max(alpha, bestVal)
            if beta <= alpha:
                logging.info(f'Pruned')
                break
        return bestVal
    else:
        bestVal = np.inf
        new_state = deepcopy(state)
        ply = optimal_strategy(new_state)
        new_state.nimming(ply)
        value = minmax(new_state, True, alpha, beta)
        bestVal = min(bestVal, value)
        return bestVal


def best_move(state: Nim):
    data = cook_status(state)
    for ply in data['possible_moves']:
        tmp_state = deepcopy(state)
        tmp_state.nimming(ply)
        score = minmax(tmp_state, my_turn=False)
        if score > 0:
            break
    return ply


# %% Q4 - RL

"""
Reinforcement learning agent to play Nim

Idea:
    Play using Upper Confidence Trees (UCT), a Monte Carlo Tree Search (MCTS) algorithm, popular when trade-off between
    finding best-so-far and finding a better one

Need:
    * All possible states (TODO: sort state so that e.g. 1 1 0 == 1 0 1)
        * Init with value 0 and visits 0
    * Actions for each state (based on data)
    * Simulate function
    * Reward function

Outline:
    1. Selection (select an unvisited node) with highest UCT
    2. Expand to that node
    3. Simulate from that node until termination
    4. Backpropagate and update node with statistics
        * N(v) - number of visits for node v
        * Q(v) - value/reward playing from that node

UCT:
    uct(v_i, v) = Q(v_i)/N(v_i) + c*sqrt(log(N(v))/N(v_i)), which prefers child nodes with small N(v_i)
    choose action according to highest uct value (init with np.inf to explore every move)
"""

# Imports
import itertools


# Class

class RLAgent:

    # INITIALIZATION -----------------------------------------------------------------
    def __init__(self, nim_size: int, random_factor=0.2,
                 exploration_factor=np.sqrt(2)):  # explore with 20%, exploit with 80%
        self.nim_size = nim_size
        self.current_state = None
        self.previous_state = None
        self.__init_states(nim_size)
        self.random_factor = random_factor
        self.c = exploration_factor

    def __init_states(self, nim_size: int):
        """find all possible board positions"""
        states = {}
        rows = [i * 2 + 1 for i in range(nim_size)]
        elem_ranges = list(itertools.combinations([range(n + 1) for n in rows], r=nim_size))
        all_states = list(itertools.product(*elem_ranges[0]))

        for state in all_states:
            states[state] = {}
            states[state]['visits'] = 0
            states[state]['value'] = 0
            states[state]['child_states'] = self.__init_child_states(state)
        self.states = states
        # last state is the initial board
        self.current_state = all_states[-1]
        self.states[self.current_state]['visits'] = 1

    def __init_child_states(self, state):
        """Find all states accessible from state"""
        nim = Nim(self.nim_size)
        nim._rows = list(state)
        if nim:
            data = cook_status(nim)
            children = []
            for ply in data['possible_moves']:
                tmp_nim = deepcopy(nim)
                tmp_nim.nimming(ply)
                children.append(tmp_nim.rows)
            return children

    # MCTS -----------------------------------------------------------------
    def selection(self):
        """Select next move according to highest uct score"""
        next_state = self.__state_with_highest_uct()
        return next_state

    def __state_with_highest_uct(self):
        """Move to child node with highest UCT score (depending on parent and child nodes) """
        visits_parent = self.states[self.current_state]['visits']
        best_state = None
        best_uct = -np.inf
        for child_state in self.states[self.current_state]['child_states']:
            visits_child = self.states[child_state]['visits']
            wins_child = self.states[child_state]['value']
            uct = wins_child / (visits_child + 1) + self.c * (np.log(visits_parent) / (visits_child + 1)) ** (1 / 2)
            if uct > best_uct:
                best_uct = uct
                best_state = child_state
        return best_state

    def random_selection(self):
        """Explore and move to random state"""
        next_state = random.choice(tuple(self.states[self.current_state]['child_states']))
        return next_state

    def expand(self, next_state):
        """Expand to the found next state. Return the ply that takes agent there"""
        self.previous_state = self.current_state
        self.current_state = next_state
        ply = self.__next_ply()
        return ply

    def __next_ply(self):
        """ Find ply that takes agent from previous state to current state"""
        # manipulate nim
        nim = Nim(self.nim_size)
        nim._rows = list(self.previous_state)
        data = cook_status(nim)
        ply = [ply for ply in data['possible_moves'] if data['rows'][ply[0]] - ply[1] == self.current_state[ply[0]]][0]
        return ply

    def simulate(self, opponent: Callable, n_matches: int):
        """Simulate game of nim vs opponent by letting RL agent play randomly from current state"""
        players = (opponent, pure_random)  # rl agent is second since played move to get here
        nim = Nim(self.nim_size)
        won = 0
        for match in range(n_matches):
            # forbidden stuff
            nim._rows = list(self.current_state)  # play from current state

            player = 0
            while nim:
                ply = players[player](nim)
                nim.nimming(ply)
                player = 1 - player
            if player == 0:
                won += 1

        # update results
        self.backpropagate(n_matches, won)

    def backpropagate(self, visits: int, reward: int):
        """Update results after simulating `visits` times game from current state"""
        self.states[self.current_state]['visits'] += visits
        self.states[self.current_state]['value'] += reward

    # TRAINING -----------------------------------------------------------------
    def learn_to_play(self, opponents: list, n_sims: int, n_matches: int):
        """Simulate the game from original state. For each move, simulate the outcome n_matches times.
        Keep moving until board is empty, then repeat n_sims times."""
        for opponent in opponents:
            for n in tqdm(range(n_sims), desc="Iterations, %s" %opponent.__name__):
                # always start from initial state in a new simulation
                nim = Nim(self.nim_size)
                self.current_state = nim.rows

                while nim:
                    if random.random() < self.random_factor:
                        # choose random state
                        ns = self.random_selection()
                    else:
                        ns = self.selection()
                    ply = self.expand(next_state=ns)
                    nim.nimming(ply)

                    self.simulate(opponent, n_matches)

    def get_statistics(self):
        """Print overview of number of visits and wins for a visited state"""
        info = [(k, v['value'], v['visits']) for k, v in self.states.items()]
        for state in info:
            if state[2] > 0:  # at least 1 visit
                print(f'State {state[0]}: \tvisits {state[2]} \twins {state[1]}')

    def policy(self, state: Nim) -> Nimply:
        """The policy, i.e. the next move for the current state"""
        self.current_state = state.rows
        ns = self.selection()
        ply = self.expand(next_state=ns)
        return ply





# %% MAIN
import argparse

if __name__ == '__main__':

    # VARIABLES
    NIM_SIZE = 3
    NUM_MATCHES = 100
    EVAL_MATCHES = 100

    # INPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task", dest="task", default=1,
                        help="Which task should run? Choose from 1, 2, 3 or 4.", type=int)

    args = parser.parse_args()
    print(f"Task: {args.task}")

    # ---------------------------TASK 1 - PLAYING THE OPTIMAL STRATEGY ---------------------------
    if args.task == 1:
        play_nim(optimal_strategy, optimal_strategy)
        # play the nim-sum strategy
        starting_wins = evaluate(optimal_strategy, optimal_strategy)
        print(f'Optimal strategy wins {starting_wins * 100: .0f}% when starting and {(1 - starting_wins) * 100: .0f}% when not starting.')

    # ---------------------------TASK 2 - EVOLVE AN AGENT ---------------------------
    elif args.task == 2:
        # set params
        POPULATION_SIZE = 50
        OFFSPRING_SIZE = 200
        GENERATIONS = 10
        OPPONENTS = [dumb_agent, pure_random, semi_smart, optimal_strategy]

        tournament_size = 10
        mutation_prob = 0.3

        pop = init_population()

        for gen in tqdm(range(GENERATIONS), desc='Generations'):
            calc_fitness(pop)
            offspring = create_offspring(pop, tournament_size, mutation_prob)
            pop = get_next_generation(offspring)

    # --------------------------- TASK 3 - MINMAX FUNCTION ---------------------------
    elif args.task == 3:
        import time
        start = time.time()
        play_nim(best_move, my_strategy)
        elapsed = time.time() - start
        print(f'It take {elapsed :.2f} seconds to play a game of Nim with size {NIM_SIZE}')

    # --------------------------- TASK 4 - REINFORCEMENT LEARNING ---------------------------
    elif args.task == 4:
        ITERS = 1000

        # must have run with -t 2 to have a pop
        if 'pop' in locals():
            opponents = [pure_random, semi_smart, make_strategy(pop[0]), optimal_strategy]
        else:
            opponents = [pure_random, semi_smart, optimal_strategy]

        for opponent in opponents:
            rl_agent = RLAgent(NIM_SIZE)
            rl_agent.learn_to_play([opponent], n_sims=ITERS, n_matches=NUM_MATCHES)
            evaluate(rl_agent.policy, opponent)


    else:
        print(f'Have not finished task {args.task}')

