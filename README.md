# State-Inference and Planning
This project explores planning in RL using traditional tabular planning algorithms (value iteration) and discrete state Variational Autoencoders to go from a high-d observation space to hashable states.

Overall, the algorithm doesn't work.  It's quite tractable to extract a meaningful adjacency graph over states, and to extract a meaningful value function, but planning seems to be too brittle.  Specifically, it seems that the models have a hard time alliacing states, and have very high entropy of state-action values.  I have been unable to either to build a working value-iteration based agent or to successfully encorporate value iteration into an existing PPO system.

## Relevant Notebooks
* `notebooks/State Inference - Cnn.ipynb` Shows the basic properties of state-inference
* `notebooks/Agent - CnnVae (online - lookahead).ipynb` Shows a value-iteration based algorithm navigating a grid-world task. Notably, the agent learns a good representation of the task but fails to learn a good policy.
* `notebooks/Agent - Ppo - Value Iteration.ipynb` Show a PPO agent with value iteration used to train the critic.
* `notebooks_valueiteration`: various analyses/exp;orations of value iteration by itself within grid-world tasks.