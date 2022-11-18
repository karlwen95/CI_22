# Lab 2 
## Set cover-problem using *genetic algorithms* (GA)

To solve the set cover-problem using genetic algorithms, I use two different approaches:
a $(\mu+\lambda)$- and $(\mu,\lambda)$-strategy.
The difference is in how the next generation is sampled.
In $(\mu+\lambda)$, the best individuals in the combined population survives whilst only the best individuals in the offspring survives when applying $(\mu,\lambda)$.
This results in different settings (i.e. `OFFSPRING_SIZE`) depending on the strategy.
The settings are specialized in the corresponding section, respectively.

### Common approach for both strategies
1. For both strategies, an initial population is randomized by sampling a random number of times from the state space (generated from the `problem`-function).
2. The fitness is then computed as the tuple `(-missing_elems, -duplicates`). Negative sign to identify the fittest individual as the one with maximum fitness.
3. Choose the parents using a roulette wheel where the fitness determines the size of the partition for each individual. This is done in the following way:
   1. Assume a population of $N$. The fittest individual will have $N$ partitions, the second fittest will have $N-1$ partitions. Following this pattern yields $N+(N-1)+...+2+1=\frac{(N+1)*N}{2}$ partitions (*arithmetic sum*)
   2. A random number `n` is sampled between $[0,\frac{(N+1)*N}{2}]$.
   3. Find which parent `n` refers to by checking in which "block" it falls. I.e. if $n \leq N$ the first parent is chosen (the fittest one). If $n \in [N+1,2N]$ parent number two is chosen (the second fittest). I.e. the probability to be chosen decreases with the fitness.
4. When two parents have been selected, generate offspring by cross-over and perform mutation with a small probability. Repeat `OFFSPRING_SIZE` times.
5. Choose the best individuals to constitute the new population.
6. Repeat step 2-5 `GENERATIONS` times.



### Results for $(\mu + \lambda)$ strategy
**Settings:**
POPULATION_SIZE = 50
OFFSPRING_SIZE = 30
ITERS = 100

| N    | weight |                                                                                                  
|------|--------|
| 5    | 5      |
| 10   | 10     |
| 20   | 25     | 
| 50   | 90     |  
| 100  | 213    |  
| 1000 | 3995   |
| 2000 | 9617   |  

### Results for $(\mu , \lambda)$ strategy

**Settings:**
POPULATION_SIZE = 50
OFFSPRING_SIZE = 300
ITERS = 100

| N    | weight |                                                                                                  
|------|--------|
| 5    | 5      |
| 10   | 11     |
| 20   | 29     | 
| 50   | 94     |  
| 100  | 191    |  
| 1000 | 3743   |
| 2000 | 8275   |  