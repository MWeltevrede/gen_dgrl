# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import AGENT_CLASSES
from online.behavior_policies.distributions import Categorical, FixedCategorical


class BehavioralCloning:
    def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64):
        """
        Initialize the agent.

        :param observation_space: the observation space for the environment
        :param action_space: the action space for the environment
        :param lr: the learning rate for the agent
        :param hidden_size: the size of the hidden layers for the agent
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.hidden_size = hidden_size

        self.model_base = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size, use_actor_linear=True)
        self.optimizer = torch.optim.Adam(self.model_base.parameters(), lr=self.lr)
        
        self.total_steps = 0

    def train(self):
        self.model_base.train()

    def eval(self):
        self.model_base.eval()

    def set_device(self, device):
        self.model_base.to(device)

    def eval_step(self, observation, eps=0.0):
        """
        Given an observation, return an action.

        :param observation: the observation for the environment
        :return: the action for the environment in numpy
        """
        if len(observation.shape) == 3:
            # add batch dimension
            observation = observation.unsqueeze(0)
        deterministic = eps == 0.0
        with torch.no_grad():
            actor_features = self.model_base(observation)
            dist = FixedCategorical(logits=actor_features)
            
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        return action.cpu().numpy()

    def train_step(self, observations, actions, rewards, next_observations, dones):
        """
        Update the agent given observations and actions.

        :param observations: the observations for the environment
        :param actions: the actions for the environment
        """
        # squeeze actions to [batch_size] if they are [batch_size, 1]
        if len(actions.shape) == 2:
            actions = actions.squeeze(dim=1)
            
        actor_features = self.model_base(observations)
        dist = FixedCategorical(logits=actor_features)
        action_log_probs = dist._get_log_softmax()
        
        self.optimizer.zero_grad()
        loss = F.nll_loss(action_log_probs, actions)
        loss.backward()
        self.optimizer.step()
        self.total_steps += 1
        # create stats dict
        stats = {"loss": loss.item(), "total_steps": self.total_steps}
        return stats

    def save(self, num_epochs, path):
        """
        Save the model to a given path.

        :param path: the path to save the model
        """
        save_dict = {
            "model_base_state_dict": self.model_base.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "curr_epochs": num_epochs
        }
        torch.save(save_dict, path)
        return

    def load(self, path):
        """
        Load the model from a given path.

        :param path: the path to load the model
        """
        checkpoint = torch.load(path)
        self.model_base.load_state_dict(checkpoint["model_base_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        return checkpoint["curr_epochs"]
    


class BehavioralCloningContinuous:
    def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64):
        """
        Initialize the agent.

        :param observation_space: the observation space for the environment
        :param action_space: the action space for the environment
        :param lr: the learning rate for the agent
        :param hidden_size: the size of the hidden layers for the agent
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr = lr
        self.hidden_size = hidden_size
        self.low = action_space.low.item()
        self.high = action_space.high.item()

        self.model_base = AGENT_CLASSES[agent_model](observation_space, action_space.shape[0], hidden_size, use_actor_linear=True)
        self.optimizer = torch.optim.Adam(self.model_base.parameters(), lr=self.lr)
        
        self.total_steps = 0

    def train(self):
        self.model_base.train()

    def eval(self):
        self.model_base.eval()

    def set_device(self, device):
        self.model_base.to(device)

    def unnormalise(self, x):
        # turn x from range [-1, 1] to [self.low, self.high]
        return ((x+1)/2.)*(self.high - self.low) + self.low

    def eval_step(self, observation, eps=0.0):
        """
        Given an observation, return an action.

        :param observation: the observation for the environment
        :return: the action for the environment in numpy
        """
        if len(observation.shape) == 3:
            # add batch dimension
            observation = observation.unsqueeze(0)
        deterministic = eps == 0.0
        with torch.no_grad():
            unbound_output = self.model_base(observation)
            action = torch.clip(self.unnormalise(unbound_output), self.low, self.high).squeeze(-1)


        return action.cpu().numpy()

    def train_step(self, observations, actions, rewards, next_observations, dones):
        """
        Update the agent given observations and actions.

        :param observations: the observations for the environment
        :param actions: the actions for the environment
        """
        # squeeze actions to [batch_size] if they are [batch_size, 1]
        if len(actions.shape) == 2:
            actions = actions.squeeze(dim=1)
            
        unbound_output = self.model_base(observations)
        policy_output = torch.clip(self.unnormalise(unbound_output), self.low, self.high).squeeze(-1)

        
        self.optimizer.zero_grad()
        loss = F.mse_loss(policy_output, actions.float())
        loss.backward()
        self.optimizer.step()
        self.total_steps += 1
        # create stats dict
        stats = {"loss": loss.item(), "total_steps": self.total_steps}
        return stats

    def save(self, num_epochs, path):
        """
        Save the model to a given path.

        :param path: the path to save the model
        """
        save_dict = {
            "model_base_state_dict": self.model_base.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "curr_epochs": num_epochs
        }
        torch.save(save_dict, path)
        return

    def load(self, path):
        """
        Load the model from a given path.

        :param path: the path to load the model
        """
        checkpoint = torch.load(path)
        self.model_base.load_state_dict(checkpoint["model_base_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        return checkpoint["curr_epochs"]
