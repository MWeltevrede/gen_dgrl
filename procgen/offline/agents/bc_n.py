# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, stack_module_state
import numpy as np
from utils import AGENT_CLASSES
from online.behavior_policies.distributions import Categorical, FixedCategorical


class BehavioralCloningEnsemble:
    def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64, ensemble_size=1, subtract_init=True):
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
        self.N = ensemble_size
        self.subtract_init = subtract_init

        self.model_base = AGENT_CLASSES["ensemble"](agent_model, ensemble_size, observation_space, action_space, hidden_size, use_actor_linear=False, subtract_init=subtract_init)
        self.model_dist = [Categorical(hidden_size, self.action_space) for _ in range(self.N)]
        self.optimizer = torch.optim.Adam(list(self.model_base.params.values()) + [p for m in self.model_dist for p in m.parameters()], lr=self.lr)
        
        self.total_steps = 0

    def train(self):
        [m.train() for m in self.model_base.models]
        [m.train() for m in self.model_dist]

    def eval(self):
        [m.eval() for m in self.model_base.models]
        [m.eval() for m in self.model_dist]

    def set_device(self, device):
        [m.to(device) for m in self.model_base.models]
        self.model_base.params, self.model_base.vec_buffers = stack_module_state(self.model_base.models)
        [m.to(device) for m in self.model_dist]
        if self.subtract_init:
            [m.to(device) for m in self.model_base.init_models]

    def eval_step(self, observation, eps=0.0):
        """
        Given an observation, return an action.

        :param observation: the observation for the environment
        :return: the action for the environment in numpy
        """
        deterministic = eps == 0.0
        with torch.no_grad():
            def fmodel(params, buffers, x):
                return functional_call(self.model_base.base_model, (params, buffers), x)
            
            actor_features = torch.vmap(fmodel, in_dims=(0,0,None))(self.model_base.params, self.model_base.vec_buffers, observation) 
            dists = [m(actor_features[i]) for i, m in enumerate(self.model_dist)]
            
            # take the mean over the logits
            logits = torch.stack([d._get_logits() for d in dists]).mean(dim=0)
            dist = FixedCategorical(logits=logits)
            
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

        def fmodel(params, buffers, x):
            return functional_call(self.model_base.base_model, (params, buffers), x)
            
        actor_features = torch.vmap(fmodel, in_dims=(0,0,None))(self.model_base.params, self.model_base.vec_buffers, observations)
        dists = [m(actor_features[i]) for i, m in enumerate(self.model_dist)]
        action_log_probs = torch.cat([d._get_log_softmax() for d in dists], dim=0)
        
        self.optimizer.zero_grad()
        loss = F.nll_loss(action_log_probs, actions.repeat(self.N))
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
            "model_base_params": self.model_base.params,
            "model_base_buffers": self.model_base.vec_buffers,
            "model_base_base_model": self.model_base.base_model,
            "model_dist_state_dict": [m.state_dict() for m in self.model_dist],
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "curr_epochs": num_epochs
        }
        if self.subtract_init:
            save_dict["init_model_base_state_dict"]= [m.state_dict() for m in self.model_base.init_models]
        torch.save(save_dict, path)
        return

    def load(self, path):
        """
        Load the model from a given path.

        :param path: the path to load the model
        """
        checkpoint = torch.load(path)
        self.model_base.params = checkpoint["model_base_params"]
        self.model_base.vec_buffers = checkpoint["model_base_buffers"]
        self.model_base.base_model = checkpoint["model_base_base_model"]
        if self.subtract_init:
            [m.load_state_dict(checkpoint["init_model_base_state_dict"][i]) for i,m in enumerate(self.model_base.init_models)]
        [m.load_state_dict(checkpoint["model_dist_state_dict"][i]) for i,m in enumerate(self.model_dist)]
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        return checkpoint["curr_epochs"]
