
## Click to play: 
[![Click to play](https://img.youtube.com/vi/BzX7KlJQHd0/0.jpg)](https://www.youtube.com/watch?v=BzX7KlJQHd0)
## Set-up
-Go to main.py, replace "fguan06" with your Wandb username.\
-Install the requirements.\
-Install the environment.
## Hardware Requirement

CUDA GPU(s)

## Environment Installation

To install the required dependencies, run the following command:

```
pip install -r requirements.txt
```
To install the Multi-Car Racing environment, please copy the multi_car_racing folder into the home directory. 
And then run:
```
cd multi_car_racing
pip install -e .
```

<hr/>

## Run Training Loop

To run the training process, use the following command:
```
python main.py
```

# Multi-Car Racing Gym Environment
<img width="100%" src="https://user-images.githubusercontent.com/11874191/98051650-5339d900-1e02-11eb-8b75-7f241d8687ef.gif"></img>
This repository contains `MultiCarRacing-v1` a multiplayer variant of Gym's original [`CarRacing-v0` environment](https://gym.openai.com/envs/CarRacing-v0/).

This environment is a simple multi-player continuous control task. The state consists of 96x96 pixels for each player. The per-player reward is `-0.1` every timestep and `+1000/num_tiles * (num_agents-past_visitors)/num_agents` for each tile visited. For example, in a race with two agents, the first agent to visit a tile receives a reward of `+1000/num_tiles` while the second agent to visit the tile receives a reward of `+500/num_tiles` for that tile. Each agent can only be rewarded once for visiting a particular tile. The motivation behind this reward structure is to be sufficiently dense for simple learnability of the basic driving skill while incentivising competition between the agents.

## Updates for v1

For each agent, the episode terminates if the agent is on grass, receiving a -100 reward. Collisions are penalized by -0.05 reward per time step for both agents involved. Going backward will also result in termination for the individual agent and a penalty of -100. Throttle and brake have been merged into the same input channel for simplified control. View switching functionality is added so that, when the ego agent terminates, it gets replaced by the subsequent agent and the view switches to that agent. 


## Basic Usage
After installation, the environment can be tried out by running:

```bash
python -m gym_multi_car_racing.multi_car_racing
```

This will launch a two-player variant (each player in its own window) that can be controlled via the keyboard (Player 1 via the arrow keys and Player 2 via `W`, `A`, `S`, `D`).

Let's quickly walk through how this environment can be used in your code:

```python
import gym
import gym_multi_car_racing

env = gym.make("MultiCarRacing-v1", num_agents=2, direction='CCW',
        use_random_direction=True, backwards_flag=True, h_ratio=0.25,
        use_ego_color=False)

obs = env.reset()[0]
done = np.zeros(env.num_agents)
total_reward = np.zeros(env.num_agents)

while not done.all():
  # The actions have to be of the format (num_agents,3)
  # The action format for each car is as in the CarRacing-v1 environment.
  for a in range(num_agents):
       action.append(agents[a].policy(obs[a]))

  # Similarly, the structure of this is the same as in CarRacing-v1 with an
  # additional dimension for the different agents, i.e.
  # obs is of shape (num_agents, 3, 96, 96)
  # reward is of shape (num_agents,)
  # done is a bool and info is not used (an empty dict).
  obs, reward, done, truncated, info = env.step(action)
  total_reward += reward
  env.render()

print("individual scores:", total_reward)
```

Overview of environment parameters:

| Parameter              | Type  | Description |
|------------------------| :---: |-------------|
| `num_agents`           |`int`  | Number of agents in environment (Default: `2`) |
| `direction`            |`str`  | Winding direction of the track. Can be `'CW'` or `'CCW'` (Default: `'CCW'`)|
| `use_random_direction` |`bool` | Randomize winding direction of the track. Disregards `direction` if enabled (Default: `True`). |
| `backwards_flag`       |`bool` | Shows a small flag if agent driving backwards (Default: `True`). |
| `h_ratio`              |`float`| Controls horizontal agent location in the state (Default: `0.25`) |
| `use_ego_color`        |`bool` | In each view the ego vehicle has the same color if  activated (Default: `False`). |

This environment contains the `CarRacing-v1` environment as a special case. It can be created via

```python
env = gym.make("MultiCarRacing-v1", num_agents=1, use_random_direction=False)
```

**Deprecation Warning:**  We might further simplify the environment in the future. Our current thoughts on deprecation concern the following functionalities.

* The direction related arguments (`use_random_direction` & `direction`) were initially added to make driving fairer as the agents' spawning locations were fixed. We resolved this unfairnes by randomizing the start positions of the agents instead.
* The impact of `backwards_flag` seems very little in practice.
* Similarly, it was interesting to play around with placing the agent at different horizontal locations of the observation (via `h_ratio`) but the default from `CarRacing-v0` ended up working well.
* The environment also contains some (not active) code on allowing penalization of driving backwards. We were worried that agents might go backwards to have more tiles on which they are first but it turned out not to be necessary for successful learning. 

We are interested in any feedback regarding these planned deprecations.

## Citation

If you find this environment useful, please cite our CoRL 2020 paper:

```bibtex
@inproceedings{SSG2020,
    title={Deep Latent Competition: Learning to Race Using Visual
      Control Policies in Latent Space},
    author={Wilko Schwarting and Tim Seyde and Igor Gilitschenski
      and Lucas Liebenwein and Ryan Sander and Sertac Karaman and Daniela Rus},
    booktitle={Conference on Robot Learning},
    year={2020}
}
```



## The original code bases can be found here: 

[Multi-Car-Racing-v0 Environment](https://github.com/igilitschenski/multi_car_racing)\
\
[Dreamer-v1 Source Code](https://github.com/kc-ml2/SimpleDreamer)

