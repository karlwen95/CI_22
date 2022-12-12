# Lab 3 - Playing Nim
In this lab we play Nim (instructions [here](https://en.wikipedia.org/wiki/Nim)) using the following agents

1. A rule-based system (an expert)
2. An agent playing with evolved rules
3. An agent based on minmax
4. An agent based on reinforcement learning

## Task 1 - the expert system
Playing Nim optimally have been solved mathematically where the agent always make a move to get the *nim-sum* to 0 (explained [here](https://en.wikipedia.org/wiki/Nim#Mathematical_theory)).

This strategy was implemented by prof. Squillero during the lectures and simply used to get the code running. 

## Task 2 - An agent using evolved rules
The set of rules for this task are (together with parameters to be evolved/learned):
1. if one row --> take elems until x remains (*optimal: remove all*) 
2. if even rows where one row has one elem --> take elems until $x$ remains from row $y$ (*optimal: remove all but one from row with mutiple elems*)
3. if odd rows where one row has one elem --> take elems until $x$ remains from row $y$ (*optimal: remove all from row with multiple elems*)
4. if even rows with multiple elems --> take elems from longest or shortest row until $x$ remains
5. if odd rows with multiple elems --> take elems from longest or shortest row until $x$ remains 

The optimal play for rule 1-3 is quickly learnt by the agent. 
Rules 4 and 5 are implemented to avoid making a random move as soon as there are more than one row with multiple elements.
Ideas to improve these are **highly appreciated**. 
My own idea is to create some rules with parameters for the agent to use to get itself in a situation where the optimal play is known (i.e. *rule 1-3*).

### Evolution of the agents
The agents are evolved following a $(\mu, \lambda)$-strategy with below described ingredients
1. Create population of `POPULATION_SIZE` with the same set of rules but different parameters for each rule (randomly initialized)
2. Fitness is computed and measured as the proportion of games won against opponents [`dumb_agent`, `pure_random`, `semi_smart`, `optimal_strategy`] and stored as a tuple. Note: the order could be changed, so that winning against an optimal strategy is higher valued than against a dumb agent.
3. $k$ individuals competes in a tournament where the winner becomes a parent. 
4. Perform cross-over between two parents (randomly select rules from parents) and mutate with `mutation_prob` (parameter(s) for one rule is randomized). 
5. Generate `OFFSPRING_SIZE` offspring and compute fitness.
6. Slice new population from the fittest offspring
7. Repeat step 2-6 `GENERATION` times

Common evolved rules after 10 generations with `POPULATION_SIZE`=50 and `OFFSPRING_SIZE`=200 have the following parameters  
`'rule_1': 0` (all elements should be removed, i.e. 0 elements should be left in the row),
`'rule_2': [1, 1]`  (all elements but 1 should be removed from row with multiple elements),
`'rule_3': [1, 0]`  (all elements should be removed from row with multiple elements),
`'rule_4': [1, 4]`  (all elements but 4 should be removed from the longest row),
`'rule_5': [0, 4]` (all elements but 4 should be removed from the shortest row).

This means that rules 1-3 are optimally evolved while the parameters for rule 4-5 are more difficult to evaluate.  
**Note:** If remaining elems $<$ param the agent only removes one element.

### Performance of evolved agent
The agents are playing `NUM_MATCHES` against above-mentioned opponents. 
With the settings mentioned in previous section, the agent usually wins all games against the dumb, random and semi-smart strategies while loosing all games against the optimal.

A description follows of the different strategies:  
**Dumb agent**: Always removes one element from the shortest row.  
**Pure random**: Always make a random move.  
**Semi smart**: Uses rules 1-3 with optimal parameters but else plays at random.  
**Optimal strategy**: Plays based on nim-sum strategy.

## Task 3 - Minmax strategy
I'm using a minmax strategy with $\alpha$ and $\beta$ pruning modified for the minimizing player to play by the optimal strategy to achieve spead-up.
Instead of going through all possible states the minimizer simply plays the optimal move according to the *Nim sum*.
If there is no optimal move a random move is made.
This implies a big speed-up and enables a game of Nim with `size = 5` to be played in approximately 10 minutes (without speed-up it didn't finish for multiple hours).
The complexity is still huge though, since a game with `size = 4` only takes a few seconds to end. 

The minmax agent performs well, as it should as it is exhaustive, winning against the optimal strategy if starting (except for `size=4`, where minmax agent wins if not starting).

## Task 4 - Reinforcement learning
**To be completed!**

## About running the file 
To get the evolved agent trained, run the file with flag `-t 2` which indicates `task 2` (use `-t 1` if you want task 1, basically optimal vs optimal).
`pop[0]` returns most fit agent.
The attributes `fitness` and `rules` retrieves the obvious.  
To play against the evolved agent, use function `play_nim(make_strategy(pop[0]),my_strategy)`.  
This displays a game of nim where `my_strategy` take user input to make a move. 
The possible moves are presented and the user enters the index (first value of each tuple). 

Good luck and let me know in *issues* if you beat my agent or not.   

### In collaboration with 
Erik Bengtsson  
Angelica Ferlin  
Leonor Gomez  
Mathias Schmekel  

