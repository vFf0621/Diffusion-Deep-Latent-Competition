
import argparse
import wandb
from dreamer.algorithms.dreamer import DreamerV3
from dreamer.utils.utils import load_config, get_base_directory
from simulate import simulate
import gymnasium as gym
import gym_multi_car_racing

'''
The main file initializes the multiCarRacing-v1 gym environment and uses the configuration
from ../configs file to initialize the agents. The wandb is used to put the data into graphs. 

The environment is found in the multi_car_racing folder

One of the cars uses an LSTM (red car) as the recurrent model and the other car uses a GRU (blue car)

'''


def main(config_file, baseline):
    config = load_config(config_file+".yml")
    env = gym.make("MultiCarRacing-v1", num_agents = 1)  
    obs_shape=env.observation_space.shape
    action_size = 2
    project_name = 'multi_car ' + config.algorithm

    with wandb.init(project=project_name, entity='fguan06', config=dict(config), settings=wandb.Settings(start_method="thread")):
        device = config.operation.device

        agent = None
        if config.algorithm == "dreamer-v3":
            agent = DreamerV3(0, obs_shape, action_size, dict(), device, config, LSTM=0, baseline=baseline)
        if config.parameters.load:
            agent.load_state_dict()
            
        # train the agent
        simulate([agent], gym.make("MultiCarRacing-v1", num_agents = 1), writer=dict(), num_interaction_episodes=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        default="Dreamerv3",
        help="Algorithm to run on even number agents (Default=Dreamerv3)",
    )
    
    parser.add_argument(
        "--baseline",
        type=bool,
        default=False,
        help="Runs the basline model if true. Otherwise, runs the diffusion augmented model (Default=False)"
    )
    
    print()
    
    main(parser.parse_args().agent, parser.parse_args().baseline)   
    
