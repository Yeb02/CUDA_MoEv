# MOdular EVolution

A 2 loops meta-learning black box algorithm. Can be applied to any problem, even those requiring interacting agents. Is best suited for tasks requiring memory and in-environment learning. Instances of the problem to be solved will be refered to as *trials*. Trials need not return rewards at each step, but only a *score* at the end.

This algorithm is currently being tested on quadruped motor control with sensorymotor information (petoi bittle and unitree go1), as well as on the video game Rocket League.


### Agent local rules

The agents use local learning rules to learn during their lifetime. Hebbian rules + modulated eligibility traces were supported (ABCD_ETA, or SPRAWL_PRUNE), but are not updated anymore so they may not compile. Currently, the learning rule is a modulated version of predictive coding. 
The agent's architecture is is very similar to that in the [DeMoCEvo](https://github.com/Yeb02/CUDA_DeMoCEvo) algorithm.

## The meta-learning algorithm

The meta-optimizer is akin to a genetic algorithm, but there significant differences.

*Agents* are not directly evolved, but are made of *phenotypic modules* (Module_P). A phenotypic module has a set of parameters that are modified by the agent during its lifelong learning, like Hebbian weights, or predictive coding weights, activations, modulation values, .... A phenotypic module also has a *type*, a pointer to a *genotypic module* (Module_G). A genotypic module is a set of parameters that are evolved by the meta-optimizer. Hebbian rules parameters, or initial predictive coding weights for instance. <br>
<br>
A phenotypic module belongs to only one agent, and an agent is nothing more than the set of its phenotypic modules. But several phenotypic modules, belonging to one or several agents, can have the same type. Genotypic modules are not owned nor managed by agents, but by the meta-optimizer. They are the evolved quantities of the algorithm, forming a common genomic pool. 
Genotypic modules are split into several *populations*, differing in module's hyperparameters. As of now, those parameters are the input size, output size, and number of children. All agents have the same architecture, they only differ by the values of their phenotypic and genotypic real-valued parameters. They also have the same number of genomic types from each population. 

<img align="left" width = 400 src="./diagrams/populations.png">
<img align="center" width = 300 src="./diagrams/agentsPool.png">

*In this example, there are 5 modules per population, and 4 agents. These parameters can be modified independently, typical value would be 256 for both. Here, agents have 1 module from population 1, 2 from pop. 2, 4 from pop. 3. This is not random, see next section.*

#### Agent structure

The agent's network architecture is a tree of phenotypic modules. The agent has a pointer to the root module, and an inference / learning step is performed by calling the appropriate recursive function on the root. The agent's observation and action are the input and output of the root <sub><sup>(not necessarily, as with PC both observations and actions can be either input or output)/sup></sub>. <br/>
A phenotypic module has an input activation vector, an output activation vector, and a set of children (phenotypic) modules. Leaf modules are those that do not have children. The root is at depth 0, (also called layer in the code) its children at depth 1, etc. All phenotypic modules at a certain depth have types from the same population, so they have the same hyperparameters, which includes the number of children. 

<img align="left" width = 400 src="./diagrams/agentArchitecture.png">
<img align="center" width = 300 src="./diagrams/moduleArchitecture.png">

*In this example, the agents have 3 "layers", and both the first and second layer's modules have 2 children. On the right, the internal structure of a Module_P that has 2 children. To make it a leaf, keep only the input and output blocs. When the local rule is predictive coding, the lefthand side vector predicts the mean of the righthand vector.*
<br />
<br />
## Lifelong learning loop

There are 2 imbricated evolution loops. The outermost is over evolution steps, called *module cycles*. The innermost is over *agents cycles*, to refine the fitness estimates.

### *agent cycle*

- Supervised learning phase: All agents experience a number of trials. During a trial, each agent is paired with a teacher agent. Teacher agents are the subset that performed the best at the previous agent cycle. At each step of the trial, the observation is transmitted to both the teacher and the student agent. The teacher's action are computed, and used to supervise the student's actions. The actions sent to the trial are those of the teacher. At the end of the trial, the score is discarded, and the pair disbanded.

- Unsupervised evaluation phase: All agents experience trials. After each trial, the agents lifetime fitnesses are updated with and exponential moving average, using a transformation of the score on the trial.

- Once all agents have been evaluated: Each agent updates the fitness of all genotypic modules that are pointed to by its phenotype (exponential moving average with the agent's fitness). Then, the agent's lifetime fitnesses are sorted, and the worst $p0$ percentile is eliminated (x is a hyperparameter). The same number of agents is created to replace them. At an agent's creation, a set of genotypic modules are sampled from the populations. The phenotype is initialized, by copying the genotype's weights to the phenotype in the case of PC. The modification of these weights during the agents lifetime does not impact the genotype: the meta-learned weights are just used to initialize the lifetime weights.

### *module cycle*, or evolution step

- Perform a number of agent cycles.
- In each population, remove the worst $p1$ percentile of genotypic modules, and replace them with new ones. A module (G) is created by first combining a *primary* parent module with its relatives, and then mutating the result. The primary parent is sampled from the population with a bias towards high fitnesses. The relatives are sampled among modules that are phylogenetically not too far from the primary parent, also with a bias towards high fitnesses. Track is kept of these relations with a phylogenetic tree. The *combination* of those modules is handled by the usual crossover operations, generalized to *n* parents (this whole process is generalized mating).

 <br />
  <br />
   <br />

Evolution then consists in repeating module cycles until convergence or satisfactory results.


# Implementation note

As of now (and despite the name), CUDA and Cublas functions are excluded from the project, since the time they take to develop and iterate on is not worth the performance gain on my hardware (GTX 1050M). Instead, acceleration relies on CPU threading and Eigen for BLAS. 

Libtorch is in the includes list but is not used. It can be removed without consequences.

There are many hyperparameters to play with, and variants of the algorithm to switch to. These can be tweaked in main.cpp and config.h .
