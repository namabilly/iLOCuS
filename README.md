# iLOCuS
Incentivizes vehicles to optimize sensing distribution in crowd sensing system

# Documentation

- [RL](#rl)  
- [Reaction](#reaction)  
- [Basics](#basics)


<a name="rl"></a>
## RL
The reinforcement learning standard core code  
- [agent](#rl-agent)
- [core](#rl-core)
- [environment](#rl-environment)
- [model](#rl-model)
- [objectives](#rl-objectives)
- [policy](#rl-policy)

<a name="rl-agent"></a>
### `agent.py`
The agent for RL
- `calc_q_value`
  - Given a state (or batch of states) calculate the Q-values.
- `update_policy`
  - Update your policy.
- `fit`
  - Fit your model to the provided environment.
- `evaluate`
  - Test your agent with a provided environment.

<a name="rl-core"></a>
### `core.py`
The core classes needed for RL
- `Sample`
  - Represents a reinforcement learning sample.
    Used to store observed experience from an MDP. Represents a
    standard `(s, a, r, s', terminal)` tuple.
- `ReplayMemory`
  - Interface for replay memories.

<a name="rl-environment"></a>
### `environment.py`
The environment for training
- `step`
  - Given an action, compute the next state and reward, i.e. progress the state.
- `_compute_reawrd`
  - Compute the reward given the current distribution of taxis and desired distribution (KL divergence).

<a name="rl-model"></a>
### `model.py`
Standard CNN, with activtion fucntion relu

<a name="rl-objectives"></a>
### `objectives.py`
The loss functions, calculating mean huber loss

<a name="rl-policy"></a>
### `policy.py`
RL Policy classes, we are using `LinearDecayGreedyEpsilonPolicy`


<a name="reaction"></a>
## Reaction
- [distance](#reaction-distance)
- [drivers](#reaction-drivers)

<a name="reaction-distance"></a>
### `distance.py`
Utility class to calculate distance and paths from index values

<a name="reaction-drivers"></a>
### `drivers.py`
Simulator for driver reactions
- `step`
  - Lottery pick to match requests and drivers. If not assigned, go to the best possible
  adjacent grid, or remain unmoved. 

<a name="basics"></a>
## Basics
Basic structures and utilities to visualize data
- [crowdsourcer](#basics-crowdsourcer)
- [model](#basics-model)
- [view](#basics-view)

<a name="basics-crowdsourcer"></a>
### `crowdsourcer.py`
The ultimate agent to make decisions, place holder

<a name="basics-model"></a>
### `model.py`
Environment simulator, provides simulated data from real data

<a name="basics"></a>
### `view.py`
Visualize data on canvas, helpful to see distributions








