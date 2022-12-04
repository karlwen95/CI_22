"""
Class for the evolvable agent.
"""

# %% imports
import random
import logging

logging.getLogger().setLevel(logging.DEBUG)

# %% CLASS

class Evolvable_agent:

    def __init__(self, nim_size):
        self._rules = {}
        # required for rules
        max_leaveInRow = (nim_size - 1) * 2

        # setting for rule 1
        self._rules['rule_1'] = random.randint(0, max_leaveInRow)
        # setting for rule 2
        self._rules['rule_2'] = [random.randint(0, 1), random.randint(0, max_leaveInRow)]
        # setting for rule 3
        self._rules['rule_3'] = [random.randint(0, 1), random.randint(0, max_leaveInRow)]
        # setting for rule 4
        self._rules['rule_4'] = [random.randint(0, 1), random.randint(0, max_leaveInRow)]
        # setting for rule 5
        self._rules['rule_5'] = [random.randint(0, 1), random.randint(0, max_leaveInRow)]

        # more rules...?

        # init fitness
        self._fitness = None

    @property
    def rules(self):
        return self._rules

    @rules.setter
    def rules(self, new_rules: dict):
        self._rules = new_rules

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, f: tuple):
        # validation if wanted
        self._fitness = f

    def rule1(self, data: dict):
        """ If only one row remains"""
        if max([m[1] for m in data['possible_moves']]) <= self._rules['rule_1']:
            move = (data['active_rows_index'][0], 1)
        else:
            tmp_move = [(r, e) for (r, e) in data['possible_moves'] if data['rows'][r] - e == self._rules['rule_1']][
                0]
            move = (tmp_move[0], tmp_move[1])
        return move

    def rule2(self, data: dict):
        """If even rows remain with one element in only one row"""
        if self._rules['rule_2'][0] == 0:  # choose from row with single elem
            row = data['single_elem_rows_index'][0]
            elem = 1  # exists only one elem to remove
            move = (row, elem)
        else:  # choose from row with multiple elems
            row = [i for i in data['active_rows_index'] if i not in data['single_elem_rows_index']][0]
            elem = max(data['rows'][row] - self._rules['rule_2'][1], 1)
            move = row, elem
        return move

    def rule3(self, data: dict):
        """If odd rows remain with one element in only one row"""
        if self._rules['rule_3'][0] == 0:  # choose from row with single elem
            row = data['single_elem_rows_index'][0]
            elem = 1  # exists only one elem to remove
            move = (row, elem)
        else:  # choose from row with multiple elems
            row = [i for i in data['active_rows_index'] if i not in data['single_elem_rows_index']][0]
            elem = max(data['rows'][row] - self._rules['rule_3'][1], 1)
            move = row, elem
        return move

    def rule4(self, data: dict):
        """If even rows remain with multiple elems in both. Choose from either longest or shortest."""
        if self.rules['rule_4'][0] == 0:  # choose from row with fewest elements
            row = data['shortest_row']
        else:
            row = data['longest_row']
        elem = max(data['rows'][row] - self.rules['rule_4'][1], 1)
        return row, elem

    def rule5(self, data: dict):
        """If odd rows remain with multiple elems in both. Choose from either longest or shortest."""
        if self.rules['rule_5'][0] == 0:  # choose from row with fewest elements
            row = data['shortest_row']
        else:
            row = data['longest_row']
        elem = max(data['rows'][row] - self.rules['rule_5'][1], 1)
        return row, elem

    def rule6(self, data: dict):
        """Play at random"""
        move = random.choice(data['possible_moves'])
        return move[0], move[1]
