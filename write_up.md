# State Inference Model

Here, we'll define two models: a *world model* and a *recognition model*.  The world model describe an enviornment and the recognition model describes the agent's inference over that enviornment. 

## World Model
We assume the envionrment can be described by a Markov Decision Process (MDP), defined by
* a set of states $s\in\mathcal{S}$
* a set of actions $a\in \mathcal{A}$
* a set of observable vectors $o\in \mathbb{R}^{d}$
* a state-action-state transtion function $\mathcal{T}: \mathcal{S}\times\mathcal{A}\to \mathcal{S}$
* a reward function over states $\mathcal{R}: \mathcal{S} \to \mathbb{R}$
* an observation function $\mathcal{O}: \mathcal{S}\to \mathbb{R}^d$

We assume that the agent knows what actions are available to it (i.e., it has knowledge of the elements $\mathcal{A}$), and can make observations of observations $o$ and rewards $r$.  Importantly, we do not assume the agent has any knowledge of the set of states.

It is helpful to define the world model probabilistically, such that our transition, reward, and observation functions, resepctively are defined as
$$\mathcal{T} = p(s'|s,a)$$
$$\mathcal{R} = p(r | s)$$
$$\mathcal{O} = p(o|s)$$

Without loss of generality, we can restrict our analysis by ignoring (or maganalizing over) the agent's policy.  This simplifies our analysis, and it is straightforward to extend it to include actions. 


## Recognition Model
Our recognition model is an apporixmate inverse of the world model. We will make assumptions to make this inference tractable, and the goal is to develop an algorithm that can infere the state of the enviornment given a history of observations.  State inference is the key computational challenge.  With known states, it is striaght forward to to approximate $\mathcal{T}$, $\mathcal{R}$, and $\mathcal{O}$ with supervised methods.

It's also worth noting that this recognition model does not generate a policy or any actions: it is strictly a model that takes in observations and outputs a distribution over states via unsupervised methods. This also means that we don't need to consider actions in the analysis.

### Approximate Model
As a further simplifying assumption, we will consider only how states depend on the observable vectors $o$, and disregard the impact of reward on inference [^1].  Our world model thus has the following graphical structure:

$$
\begin{array}{ccccccc}
s_{t} & \to & s_{t+1} & \to & s_{t+2} & \to & \dots \\
\downarrow && \downarrow && \downarrow \\ 
o_{t} &  & o_{t+1} &  & o_{t+2} &  & \dots
\end{array}
$$

Given this structure, we can define a distribution over states $s_t$ at time $t$ recursively as
$$
p(s_t|o_t, s_{t-1}) = p(s_t| o_t)p(s_t|s_{t-1})
$$


Thus, our state-estimation problem has been broken down into two sub-problem: (1) estimating a distribution over states from observations and (2) estimating state-to-state transition functions. The goal is to use a parameterized function to estimate both.  Before we delve into how we estimate these sub-problems, its helpful to state representation.  


### Representation of States
A key computational goal is to invert the observation function $\mathcal{O}$ using a parameterized function. More formally, we want to learn the inverse of the observation function $\mathcal{O}^{-1}(o)  = s$.  The agent never observes $s$, so this inverse is not learnable.  

Instead, we will assume that our observations can be described as a part of a latent variable model with the joint distribution $p(o, z)$.  We will assume a bijective mapping  between states $\mathcal{S}$ and latent variables $\mathcal{Z}$.[^2] While this assumption is likely false, it allows us to rediscribe our world models in terms of $z$ and maintain the Markov property.

Thus, this transforms our approximate inference problem into learning the distribution over $z$: 

$$
\begin{equation}
p(z_t|o_t, z_{t-1}) = p(z_t| o_t)p(z_t|z_{t-1})
\end{equation}
$$


The set $\mathcal{Z}$ is an assumption of the recognition model and we can define it's properties.  We want to assume $\mathcal{Z}$ contains a finite, but arbitrarily large, number of states that approximately independent.  We want to be able to encode $\mathcal{Z}$ in a vector spaces so that it is learnable with a parameterize function.


A useful representation of states that satisisfies these desiderata is then a set of one-hot vectors.  Let $$z=\left [z_1,.., z_n\right ]$$ be a set of $n$ on hot vectors, each of which is length $b$, or $z_{i}\in \{0, 1\}^b$, $\sum_{j=1}^b z_{i,j} = 1$. This representation can represent $b^n$ states and it can be shown that random vectors are likely independent for sufficiently large values of $b$.[^3]


In equation (1) above, there are two terms, the inverse observation model $p(z_t| o_t)$ and the transition model $p(z_t|z_{t-1})$.  Below, we first describe the observation model $p(z_t|o_t)$ before returning to a discussion of the transition model.

### Observation model
To make inference tractible, we apply the mean-field assumption to our latent variables[^4], such that 

$$p(z|o)=\prod p(z_i|o)$$

The probability of each one-hot vector $z_i$ is defined with the categorical distribution 
$$
p(z_{ij}) =  \frac{\omega_{ij} }{\sum_{b}^{j=1}\omega_{ij}}
$$

where $\log \omega_{i} = f_\theta(o)$ is the output of a neural network.[^5] In practice, it is convient to work directly with $\log \omega_{i}$.

### Transition function

Here, we have a couple of options.  One posibility is to work with a MAP latent-varable estimate $\hat{z_i}$, and estimate a tabular transition function estimate.




***

### Footnotes

[^1]: Alternatively, we could inlude the reward signal in our observation vector by creating a new variable. E.g. $o'=[r, o]$

[^2]: A bijective mapping is a one-to-one and intertable mapping.


[^3]: There is a tradeoff between the independence of the state variables and the number of states we can represent. If we assume a fixed representational capacity $k$ such that $b\times n=k$, then the number of states we can represent is maximized when $b=2$ and $n=k/2$ (i.e. a binary string of length $k/2$), and the states are most independent when $b=k$ and $n=1$ (i.e. a single one-hot vector of length $k$).  The binary string can has a much larger representational capacity, as it can represesent $2^{k/2}$ states vs the $k$ states of a single one-hot vector.  In contrast, two random binary vectors are likely to be similar: if $x$ and $y$ are random binary vectors of length $k$, then their expected dot-product is $\mathbb{E}[x^Ty] = 0.25\times (k/2)$. In contrast, if $x$ and $y$ are one-hot vectors of length $k$, then their expect dot-product is $\mathbb{E}[x^Ty]=\frac{1}{k}$. 

[^4] As an alternative, we can imagine a case where we define the latent variables auto-regresively, such that $p(z_i) = \prod_{j=1}^{i-1}p(z_j,| z_{j-1}, o)$

[^5] In practice, the function $f_\theta$ is trained by drawing samples from a Gumbel-softmax parameterized with temperature $\tau\in (0, \infty)$.  For inference, we can assume $\tau=0$, which reduces the Gumbel-softmax distribution to a categorical distribution. 