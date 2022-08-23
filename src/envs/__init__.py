from functools import partial
from types import SimpleNamespace
from .multiagentenv import MultiAgentEnv
from onpolicy.envs.mpe import MPE_env
from onpolicy.envs.football import Football_Env
# from .starcraft2.starcraft2 import StarCraft2Env
import sys
import os
import numpy as np
import torch

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

class MPE(MultiAgentEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        
        args = SimpleNamespace(**kwargs)
        self.env = MPE_env.MPEEnv(args)

        self.episode_limit = self.env.world_length
        self.n_agents = len(self.env.agents)

        self.n_actions = self.env.action_space[0].n

    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        obs, reward, done, info = self.env.step(actions)
        reward = np.array(reward)
        return reward.mean(), np.array(done).any(), {}
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
    
    def get_state_size(self):
        return self.env.share_observation_space[0].shape[0]
    
    def get_obs_size(self):
        return self.env.observation_space[0].shape[0]

    def get_total_actions(self):
        return self.n_actions
    
    def get_state(self):
        obs = self.get_obs()
        return obs.flatten()

    def get_obs(self):
        obs_n = []
        for i, agent in enumerate(self.env.agents):
            obs_n.append(self.env._get_obs(agent))
        return np.stack(obs_n)
        
    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions))

class Football(MultiAgentEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.env = Football_Env.FootballEnv(SimpleNamespace(**kwargs))

        self.episode_limit = self.env.max_steps
        self.n_agents = self.env.num_agents
        self.n_actions = self.env.action_space[0].n
    
    def reset(self):
        return self.env.reset()
    
    def step(self, actions):
        self.obs, reward, done, info = self.env.step(actions)
        return reward.mean(), done, info
    
    def get_state_size(self):
        return self.env.observation_space[0].shape[0] * self.n_agents
    
    def get_state(self):
        return self.obs.flatten()
    
    def get_obs(self):
        return self.obs
    
    def get_avail_actions(self):
        return np.ones((self.n_agents, self.n_actions))

    
REGISTRY["mpe"] = partial(env_fn, env=MPE)
REGISTRY["football"] = partial(env_fn, Football)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
