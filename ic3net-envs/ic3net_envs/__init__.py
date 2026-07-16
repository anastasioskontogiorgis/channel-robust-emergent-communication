# from gym.envs.registration import register

# register(
#     id='PredatorPrey-v0',
#     entry_point='ic3net_envs.predator_prey_env:PredatorPreyEnv',
# )

# register(
#     id='TrafficJunction-v0',
#     entry_point='ic3net_envs.traffic_junction_env:TrafficJunctionEnv',
# )

import gym
from gym.envs.registration import register

# get already refgistered enviroments 
env_dict = gym.envs.registration.registry.env_specs.copy()

# if already registered remove - issue with multiprocessing 
for env in env_dict:
    if 'TrafficJunction-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]

# register(
#     id='PredatorPrey-v0',
#     entry_point='ic3net_envs.predator_prey_env:PredatorPreyEnv',
# )

register(
    id='TrafficJunction-v0',
    entry_point='ic3net_envs.traffic_junction_env:TrafficJunctionEnv',
)