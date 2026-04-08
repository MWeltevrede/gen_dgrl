# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from copy import deepcopy

import wandb
from online.behavior_policies.distributions import Categorical, Normal
from utils.augmentations import rotate, identity, crop, random_conv, color_jitter
from utils import AGENT_CLASSES
from gym import spaces


class IQL:
	def __init__(
		self,
		observation_space,
		action_space,
		lr,
		agent_model,
		hidden_size,
		channels,
		gamma,
		target_update_freq,
		tau,
		eps_start,
		eps_end,
		eps_decay,
		iql_temperature,
		iql_expectile,
		perform_polyak_update, 
		normalize_obs,
		activation,
	):
		"""
		Initialize the agent.

		:param observation_space: the observation space for the environment
		:param action_space: the action space for the environment
		:param lr: the learning rate for the agent
		:param hidden_size: the size of the hidden layers for the agent
		:param gamma: the discount factor for the agent
		:param target_update_freq: the frequency with which to update the target network
		:param tau: the soft update factor for the target network
		:param eps_start: the starting epsilon for the agent
		:param eps_end: the ending epsilon for the agent
		:param eps_decay: the decay rate for epsilon
		:param iql_temperature: the temperature for the IQL agent
		:param iql_expectile: the expectile weight for the IQL agent
		"""

		# Implement Implicit Q Learning, which has an Actor, Critic and Q Function
		self.observation_space = observation_space
		self.lr = lr
		self.hidden_size = hidden_size
		self.gamma = gamma
		self.target_update_freq = target_update_freq
		self.tau = tau
		self.agent_model = agent_model
		self.channels = channels
		self.normalize_obs = normalize_obs
		self.activation = activation

		self.total_steps = 0

		self.eps_start = eps_start
		self.eps_end = eps_end
		self.eps_decay = eps_decay

		self.iql_temperature = iql_temperature
		self.iql_expectile = iql_expectile
		
		self.perform_polyak_update = perform_polyak_update

		# Implement the Actor, Critic and Value Function
		if isinstance(action_space, spaces.Discrete):
			self.action_space = action_space.n
			self.actor_dist = Categorical(hidden_size, self.action_space)
			self.continuous_actions = False
			self.action_space_actor = action_space.n
		elif isinstance(action_space, spaces.Box):
			self.action_space = action_space.shape[0]
			self.low = torch.as_tensor(action_space.low).float()
			self.high = torch.as_tensor(action_space.high).float()
			self.actor_dist = Normal(hidden_size, self.action_space)
			assert isinstance(observation_space, spaces.Box), "Only Box observation space is supported for continuous action space."
			self.action_observation_space = spaces.Box(np.array([*observation_space.low, *action_space.low]),np.array([*observation_space.high, *action_space.high]), dtype=observation_space.dtype)
			self.continuous_actions = True
			self.action_space_actor = 2*action_space.shape[0]

		self.model_actor = AGENT_CLASSES[agent_model](
			observation_space, self.action_space_actor, hidden_size, channels, use_actor_linear=True, normalize_obs=self.normalize_obs, activation=self.activation,
		)
		# optimizer_actor uses parameters from model_actor and actor_dist
		actor_model_params = list(self.model_actor.parameters()) 
		self.optimizer_actor = torch.optim.Adam(actor_model_params, lr=self.lr)

		self.model_v = AGENT_CLASSES[agent_model](observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
		self.optimizer_v = torch.optim.Adam(self.model_v.parameters(), lr=self.lr)

		if self.continuous_actions:
			self.model_q1 = AGENT_CLASSES[agent_model](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
			self.target_q1 = AGENT_CLASSES[agent_model](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
		else:
			self.model_q1 = AGENT_CLASSES[agent_model](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
			self.target_q1 = AGENT_CLASSES[agent_model](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
		self.optimizer_q1 = torch.optim.Adam(self.model_q1.parameters(), lr=self.lr)
		self.target_q1.load_state_dict(self.model_q1.state_dict())
		self.target_q1.eval()

		if self.continuous_actions:
			self.model_q2 = AGENT_CLASSES[agent_model](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
			self.target_q2 = AGENT_CLASSES[agent_model](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
		else:
			self.model_q2 = AGENT_CLASSES[agent_model](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
			self.target_q2 = AGENT_CLASSES[agent_model](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
		self.optimizer_q2 = torch.optim.Adam(self.model_q2.parameters(), lr=self.lr)
		self.target_q2.load_state_dict(self.model_q2.state_dict())
		self.target_q2.eval()

	def train(self):
		self.model_actor.train()
		self.actor_dist.train()
		self.model_v.train()
		self.model_q1.train()
		self.model_q2.train()

	def eval(self):
		self.model_actor.eval()
		self.actor_dist.eval()
		self.model_v.eval()
		self.model_q1.eval()
		self.model_q2.eval()

	def reset_actor(self):
		if not self.continuous_actions:
			self.actor_dist = Categorical(self.hidden_size, self.action_space)
		else:
			self.actor_dist = Normal(self.hidden_size, self.action_space)

		self.model_actor = AGENT_CLASSES[self.agent_model](
			self.observation_space, 2*self.action_space, self.hidden_size, self.channels, use_actor_linear=True, normalize_obs=self.normalize_obs, activation=self.activation,
		)
		# optimizer_actor uses parameters from model_actor and actor_dist
		actor_model_params = list(self.model_actor.parameters())
		self.optimizer_actor = torch.optim.Adam(actor_model_params, lr=self.lr)

		self.model_actor.to(self.device)
		self.actor_dist.to(self.device)

	def set_device(self, device):
		self.model_actor.to(device)
		self.actor_dist.to(device)
		self.model_v.to(device)
		self.model_q1.to(device)
		self.model_q2.to(device)
		self.target_q1.to(device)
		self.target_q2.to(device)
		self.device = device
		if self.continuous_actions:
			self.low = self.low.to(device)
			self.high = self.high.to(device)

	def expectile_loss(self, u_diff, expectile=0.8):
		"""
		Calculate the expectile loss for the IQL agent.

		:param value: the value function shape [batch_size, 1]
		:param Q_value: the Q-value shape [batch_size, 1]
		:param expectile: the expectile weight
		"""
		# expectile_weight = torch.where(u_diff > 0, expectile, 1 - expectile)  # [batch_size, 1]
		# L2_tau = expectile_weight * (u_diff**2)  # [batch_size, 1]
		return torch.mean(torch.abs(expectile - (u_diff < 0).float()) * u_diff**2)

	def train_step(self, observations, actions, rewards, next_observations, dones):
		# 1. Calculate Value Loss
		with torch.no_grad():
			if self.continuous_actions:
				q1 = self.target_q1(torch.concat([observations, actions], dim=-1))  # [batch_size, 1]
				q2 = self.target_q2(torch.concat([observations, actions], dim=-1))  # [batch_size, 1]
			else:
				q1 = self.target_q1(observations).gather(1, actions)  # [batch_size, 1]
				q2 = self.target_q2(observations).gather(1, actions)  # [batch_size, 1]
			q_minimum = torch.min(q1, q2)  # [batch_size, 1]

		curr_value = self.model_v(observations)  # [batch_size, 1]
		u_diff = q_minimum - curr_value  # [batch_size, 1]
		value_loss = self.expectile_loss(u_diff, self.iql_expectile)  # [1]
		self.optimizer_v.zero_grad(set_to_none=True)
		value_loss.backward()
		self.optimizer_v.step()

		# 2. Calculate Critic Loss
		with torch.no_grad():
			next_v = self.model_v(next_observations)  # [batch_size, 1]
		target_q = rewards + (1 - dones) * self.gamma * next_v.detach()  # [batch_size, 1]
		if self.continuous_actions:
			curr_q1 = self.model_q1(torch.concat([observations, actions], dim=-1))  # [batch_size, 1]
			curr_q2 = self.model_q2(torch.concat([observations, actions], dim=-1))  # [batch_size, 1]
		else:
			curr_q1 = self.model_q1(observations).gather(1, actions)  # [batch_size, 1]
			curr_q2 = self.model_q2(observations).gather(1, actions)  # [batch_size, 1]
		critic1_loss = F.mse_loss(curr_q1, target_q).mean()  # [1]
		self.optimizer_q1.zero_grad(set_to_none=True)
		critic1_loss.backward()
		self.optimizer_q1.step()

		critic2_loss = F.mse_loss(curr_q2, target_q).mean()  # [1]
		self.optimizer_q2.zero_grad(set_to_none=True)
		critic2_loss.backward()
		self.optimizer_q2.step()
		
		# Update the target network, copying all weights and biases in DQN
		if self.perform_polyak_update:
			for target_param, param in zip(self.target_q1.parameters(), self.model_q1.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for target_param, param in zip(self.target_q2.parameters(), self.model_q2.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		else:
			if self.total_steps % self.target_update_freq == 0:
				self.target_q1.load_state_dict(self.model_q1.state_dict())
				self.target_q2.load_state_dict(self.model_q2.state_dict())

		# 3. Calculate Actor Loss
		exp_action = torch.exp(u_diff.detach() * self.iql_temperature)  # [batch_size, 1]
		# take minimum of exp_action and 100.0 to avoid overflow
		exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
		# _, action_log_prob = self.get_action(observations, return_log_probs=True)  # [batch_size, 1]
		action_feats = self.model_actor(observations)  
		action_dist = self.actor_dist(action_feats)
		if self.continuous_actions:
			action_log_prob = action_dist.log_probs(self.normalise(actions))
		else:
			action_log_prob = action_dist.log_probs(actions)
		actor_loss = -(exp_action * action_log_prob).mean()  # [1]
		self.optimizer_actor.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.optimizer_actor.step()

		self.total_steps += 1

		# create stats dict
		with torch.no_grad():
			loss = value_loss + actor_loss + critic1_loss + critic2_loss
		stats = {
			"loss": loss.item(),
			"value_loss": value_loss.item(),
			"critic1_loss": critic1_loss.item(),
			"critic2_loss": critic2_loss.item(),
			"actor_loss": actor_loss.item(),
			"total_steps": self.total_steps,
		}
		# print(stats["actor_loss"])
		return stats
	
	def train_actor(self, observations, actions, rewards, next_observations, dones):
		with torch.no_grad():
			if self.continuous_actions:
				q1 = self.target_q1(torch.concat([observations, actions], dim=-1))  # [batch_size, 1]
				q2 = self.target_q2(torch.concat([observations, actions], dim=-1))  # [batch_size, 1]
			else:
				q1 = self.target_q1(observations).gather(1, actions)  # [batch_size, 1]
				q2 = self.target_q2(observations).gather(1, actions)  # [batch_size, 1]
			q_minimum = torch.min(q1, q2)  # [batch_size, 1]
			curr_value = self.model_v(observations)  # [batch_size, 1]
			u_diff = q_minimum - curr_value  # [batch_size, 1]

		exp_action = torch.exp(u_diff.detach() * self.iql_temperature)  # [batch_size, 1]
		# take minimum of exp_action and 100.0 to avoid overflow
		exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
		# _, action_log_prob = self.get_action(observations, return_log_probs=True)  # [batch_size, 1]
		action_feats = self.model_actor(observations)  
		action_dist = self.actor_dist(action_feats)
		if self.continuous_actions:
			action_log_prob = action_dist.log_probs(self.normalise(actions))
		else:
			action_log_prob = action_dist.log_probs(actions)
		actor_loss = -(exp_action * action_log_prob).mean()  # [1]
		self.optimizer_actor.zero_grad(set_to_none=True)
		actor_loss.backward()
		self.optimizer_actor.step()

		self.total_steps += 1

		# create stats dict
		stats = {
			"actor_actor_loss": actor_loss.item(),
			"actor_total_steps": self.total_steps,
		}
		# print(stats["actor_loss"])
		return stats

	@property
	def calculate_eps(self):
		"""
		Calculate epsilon given the current timestep, initial epsilon, end epsilon and decay rate.
		"""
		eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1.0 * self.total_steps / self.eps_decay)
		return eps
	
	def unnormalise(self, x):
		# turn x from range [-1, 1] to [self.low, self.high]
		x = torch.tanh(x)
		return ((x+1)/2.)*(self.high - self.low) + self.low
	
	def normalise(self, x):
		# turn x from range [self.low, self.high] to [-1, 1]
		x = (2*(x - self.low) / (self.high - self.low)) - 1
		return torch.atanh(x)

	def eval_step(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0
		with torch.no_grad():
			action_feats = self.model_actor(observations)
			action_dist = self.actor_dist(action_feats)

			if deterministic:
				action = action_dist.mode()
			else:
				action = action_dist.sample()  # [batch_size, 1]

			action_log_prob = action_dist.log_probs(action)

		if self.continuous_actions:
			action = self.unnormalise(action)

		if return_log_probs:
			return action.cpu().numpy(), action_log_prob

		return action.cpu().numpy()

	def get_action(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0

		action_feats = self.model_actor(observations)  
		action_dist = self.actor_dist(action_feats)

		if deterministic:
			action = action_dist.mode()
		else:
			action = action_dist.sample()  # [batch_size, 1]

		action_log_prob = action_dist.log_probs(action)
		# st()
		# print(action_log_prob)

		if self.continuous_actions:
			action = self.unnormalise(action)

		if return_log_probs:
			return action, action_log_prob

		return action

	def save(self, num_epochs, path):
		"""
		Save the model to a given path.

		:param path: the path to save the model
		"""
		save_dict = {
			"actor_state_dict": self.model_actor.state_dict(),
			#"actor_dist_state_dict": self.actor_dist.state_dict(),
			"model_v_state_dict": self.model_v.state_dict(),
			"model_q1_state_dict": self.model_q1.state_dict(),
			"model_q2_state_dict": self.model_q2.state_dict(),
			"target_q1_state_dict": self.target_q1.state_dict(),
			"target_q2_state_dict": self.target_q2.state_dict(),
			"optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
			"optimizer_v_state_dict": self.optimizer_v.state_dict(),
			"optimizer_q1_state_dict": self.optimizer_q1.state_dict(),
			"optimizer_q2_state_dict": self.optimizer_q2.state_dict(),
			"total_steps": self.total_steps,
			"curr_epochs": num_epochs,
		}
		torch.save(save_dict, path)
		return

	def load(self, path):
		"""
		Load the model from a given path.

		:param path: the path to load the model
		"""
		checkpoint = torch.load(path)
		self.model_actor.load_state_dict(checkpoint["actor_state_dict"])
		#self.actor_dist.load_state_dict(checkpoint["actor_dist_state_dict"])
		self.model_v.load_state_dict(checkpoint["model_v_state_dict"])
		self.model_q1.load_state_dict(checkpoint["model_q1_state_dict"])
		self.model_q2.load_state_dict(checkpoint["model_q2_state_dict"])
		self.target_q1.load_state_dict(checkpoint["target_q1_state_dict"])
		self.target_q2.load_state_dict(checkpoint["target_q2_state_dict"])
		self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
		self.optimizer_v.load_state_dict(checkpoint["optimizer_v_state_dict"])
		self.optimizer_q1.load_state_dict(checkpoint["optimizer_q1_state_dict"])
		self.optimizer_q2.load_state_dict(checkpoint["optimizer_q2_state_dict"])
		self.total_steps = checkpoint["total_steps"]
		self.target_q1.eval()
		self.target_q2.eval()

		return checkpoint["curr_epochs"]
	

C4 = [0, 90, 180, 270]

class IQLEnsemble(IQL):
	def __init__(
		self,
		observation_space,
		action_space,
		lr,
		agent_model,
		hidden_size,
		channels,
		gamma,
		target_update_freq,
		tau,
		eps_start,
		eps_end,
		eps_decay,
		iql_temperature,
		iql_expectile,
		perform_polyak_update, 
		normalize_obs,
		activation,
		value_ensemble_size,
		actor_ensemble_size,
		use_value,
		critic_da,
		critic_concistency_coef,
		actor_da,
		actor_concistency_coef,
		augmentation_type,
		actor_soda_update_coef = 0.005,
		avg_q = False,
		extract_all_actions = False,
		detach_original = False,
		consistency_probs = False
	):
		super().__init__(
			observation_space=observation_space,
			action_space=action_space,
			lr=lr,
			agent_model=agent_model,
			hidden_size=hidden_size,
			channels=channels,
			gamma=gamma,
			target_update_freq=target_update_freq,
			tau=tau,
			eps_start=eps_start,
			eps_end=eps_end,
			eps_decay=eps_decay,
			iql_temperature=iql_temperature,
			iql_expectile=iql_expectile,
			perform_polyak_update=perform_polyak_update,
			normalize_obs=normalize_obs,
			activation=activation,
		)
		self.value_ensemble_size = value_ensemble_size
		self.actor_ensemble_size = actor_ensemble_size
		self.use_value = use_value
		self.avg_q = avg_q
		self.extract_all_actions = extract_all_actions
		self.agent_model = agent_model
		self.critic_da = critic_da
		self.critic_concistency_coef = critic_concistency_coef
		self.actor_da = actor_da
		self.actor_concistency_coef = actor_concistency_coef
		self.actor_soda_update_coef = actor_soda_update_coef
		self.detach_original = detach_original
		self.consistency_probs = consistency_probs
		del self.model_q1
		del self.target_q1
		del self.optimizer_q1
		del self.model_q2
		del self.target_q2
		del self.optimizer_q2
		del self.model_actor
		del self.optimizer_actor
		if augmentation_type == 'rotate':
			self.augmentation = rotate
		elif augmentation_type == 'identity':
			self.augmentation = identity
		elif augmentation_type == 'crop':
			self.augmentation = crop
		elif augmentation_type == 'random_conv':
			self.augmentation = random_conv
		elif augmentation_type == 'color_jitter':
			self.augmentation = color_jitter

		if agent_model == 'illustrative':
			self.model_actor = AGENT_CLASSES['illustrative_ensemble'](
				observation_space, self.action_space_actor, hidden_size, channels, use_actor_linear=True, normalize_obs=self.normalize_obs, activation=self.activation, ensemble_size=actor_ensemble_size,
			)
			self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=self.lr)
		else:
			assert actor_ensemble_size == 1, "Actor ensemble size > 1 is only supported for illustrative environment."
			self.model_actor = AGENT_CLASSES[agent_model](
				observation_space, self.action_space_actor, hidden_size, channels, use_actor_linear=True, normalize_obs=self.normalize_obs, activation=self.activation,
			)
			self.optimizer_actor = torch.optim.Adam(self.model_actor.parameters(), lr=self.lr)


		self.model_qs = []
		self.target_qs = []
		self.optimizer_qs = []
		if agent_model == 'illustrative':
			if self.continuous_actions:
				self.model_qs = AGENT_CLASSES['illustrative_ensemble'](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation, ensemble_size=value_ensemble_size)
				self.target_qs = AGENT_CLASSES['illustrative_ensemble'](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation, ensemble_size=value_ensemble_size)
			else:
				self.model_qs = AGENT_CLASSES['illustrative_ensemble'](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation, ensemble_size=value_ensemble_size)
				self.target_qs = AGENT_CLASSES['illustrative_ensemble'](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation, ensemble_size=value_ensemble_size)
			self.optimizer_qs = torch.optim.Adam(self.model_qs.parameters(), lr=self.lr)
			self.target_qs.load_state_dict(self.model_qs.state_dict())
			self.target_qs.eval()
		else:
			for _ in range(value_ensemble_size):
				if self.continuous_actions:
					model_q = AGENT_CLASSES[agent_model](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
					target_q = AGENT_CLASSES[agent_model](self.action_observation_space, 1, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
				else:
					model_q = AGENT_CLASSES[agent_model](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
					target_q = AGENT_CLASSES[agent_model](observation_space, self.action_space, hidden_size, channels, normalize_obs=self.normalize_obs, activation=self.activation,)
				optimizer_q = torch.optim.Adam(model_q.parameters(), lr=self.lr)
				target_q.load_state_dict(model_q.state_dict())
				target_q.eval()
				
				self.model_qs.append(model_q)
				self.target_qs.append(target_q)
				self.optimizer_qs.append(optimizer_q)

		#if self.actor_da == "concistency_soda":
		#	self.target_actor = deepcopy(self.model_actor)
		#	self.target_actor.eval()
		#	self.soda_projector = nn.Sequential(
		#		nn.Linear(hidden_size, 100),
		#		nn.ReLU(),
		#		nn.Linear(100, 100)
		#	)
		#	self.soda_projector_target = nn.Sequential(
		#		nn.Linear(hidden_size, 100),
		#		nn.ReLU(),
		#		nn.Linear(100, 100)
		#	)
		#	self.soda_projector_target.load_state_dict(self.soda_projector.state_dict())
		#	self.soda_projector_target.eval()
		#	self.soda_predictor = nn.Sequential(
		#		nn.Linear(100, 100),
		#		nn.ReLU(),
		#		nn.Linear(100, 100)
		#	)

		#	actor_model_params = list(self.model_actor.parameters()) + list(self.actor_dist.parameters()) + list(self.soda_projector.parameters()) + list(self.soda_predictor.parameters())
		#	self.optimizer_actor = torch.optim.Adam(actor_model_params, lr=self.lr)
		assert not self.actor_da == "concistency_soda"

	def train(self):
		self.model_actor.train()
		#[m.train() for m in self.model_actor]
		self.actor_dist.train()
		self.model_v.train()
		if self.agent_model == 'illustrative':
			self.model_qs.train()
		else:
			for m in self.model_qs:
				m.train()

	def eval(self):
		self.model_actor.eval()
		#[m.eval() for m in self.model_actor]
		self.actor_dist.eval()
		self.model_v.eval()
		if self.agent_model == 'illustrative':
			self.model_qs.eval()
		else:
			for m in self.model_qs:
				m.eval()

	def set_device(self, device):
		self.model_actor.to(device)
		#[m.to(device) for m in self.model_actor]
		self.actor_dist.to(device)
		self.model_v.to(device)
		if self.agent_model == 'illustrative':
			self.model_qs.to(device)
			self.target_qs.to(device)
		else:
			for m in self.model_qs:
				m.to(device)
			for t in self.target_qs:
				t.to(device)
		self.device = device
		if self.continuous_actions:
			self.low = self.low.to(device)
			self.high = self.high.to(device)

		#if self.actor_da == "concistency_soda":
		#	self.target_actor.to(device)
		#	self.soda_projector.to(device)
		#	self.soda_predictor.to(device)
		#	self.soda_projector_target.to(device)
			
	def train_step(self, observations, actions, rewards, next_observations, dones):
		# 1. Calculate Value Loss
		if "augment" in self.critic_da:
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(observations.shape[0])]
				critic_observations = torch.concat([observations, self.augmentation(observations, angles)], dim=0)
			else:
				critic_observations = torch.concat([observations, self.augmentation(observations.clone())], dim=0)
			critic_actions = torch.concat([actions, actions], dim=0)
			critic_rewards = torch.concat([rewards, rewards], dim=0)
			critic_dones = torch.concat([dones, dones], dim=0)
			if self.critic_da == "augment_both":
				if self.agent_model == 'illustrative':
					critic_next_observations = torch.concat([next_observations, self.augmentation(next_observations, angles)], dim=0)
				else:
					critic_next_observations = torch.concat([next_observations, self.augmentation(next_observations.clone())], dim=0)
			else:
				critic_next_observations = torch.concat([next_observations, next_observations], dim=0)
		else:
			critic_observations = observations
			critic_actions = actions
			critic_rewards = rewards
			critic_dones = dones
			critic_next_observations = next_observations

		with torch.no_grad():
			if self.agent_model == 'illustrative':
				if self.continuous_actions:
					qs = self.target_qs(torch.concat([critic_observations, critic_actions], dim=-1))	# [value_ensemble_size, batch_size, 1]
				else:
					qs = self.target_qs(critic_observations).gather(-1, critic_actions.unsqueeze(0).repeat_interleave(self.value_ensemble_size, dim=0))
					assert qs.shape[-1] == 1
			else:
				qs = []
				if self.continuous_actions:
					for t in self.target_qs:
						qs.append(t(torch.concat([critic_observations, critic_actions], dim=-1)))
				else:
					for t in self.target_qs:
						qs.append(t(critic_observations).gather(1, critic_actions))
				qs = torch.stack(qs, dim=0)		# [value_ensemble_size, batch_size, 1]

			#q_minimum = torch.min(qs, dim=0)[0]  # [batch_size, 1]
			q_minimum = torch.min(qs[0], qs[1])
			#q_avg = torch.mean(qs, dim=0) 	# [batch_size, 1]

		curr_value = self.model_v(critic_observations)  # [batch_size, 1]
		u_diff = q_minimum - curr_value  # [batch_size, 1]
		value_loss = self.expectile_loss(u_diff, self.iql_expectile)  # [1]

		value_concistency_loss = 0
		if self.critic_da == "concistency":
			latent = self.model_v.get_last_latent(critic_observations)
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(critic_observations.shape[0])]
				augmented_latent = self.model_v.get_last_latent(self.augmentation(critic_observations, angles))
			else:
				augmented_latent = self.model_v.get_last_latent(self.augmentation(critic_observations.clone()))
			if self.detach_original:
				latent = latent.detach()
			value_concistency_loss = self.critic_concistency_coef * F.mse_loss(latent, augmented_latent).mean()
			value_loss += value_concistency_loss
			value_concistency_loss = value_concistency_loss.item()
		elif self.critic_da == "concistency_output":
			latent = self.model_v(critic_observations)
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(critic_observations.shape[0])]
				augmented_latent = self.model_v(self.augmentation(critic_observations, angles))
			else:
				augmented_latent = self.model_v(self.augmentation(critic_observations.clone()))
			if self.detach_original:
				latent = latent.detach()
			value_concistency_loss = self.critic_concistency_coef * F.mse_loss(latent, augmented_latent).mean()
			value_loss += value_concistency_loss
			value_concistency_loss = value_concistency_loss.item()
		elif self.critic_da == "augment_concistency":
			latent = self.model_v(observations)
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(observations.shape[0])]
				augmented_latent = self.model_v(self.augmentation(observations, angles))
			else:
				augmented_latent = self.model_v(self.augmentation(observations.clone()))
			if self.detach_original:
				latent = latent.detach()
			value_concistency_loss = self.critic_concistency_coef * F.mse_loss(latent, augmented_latent).mean()
			value_loss += value_concistency_loss
			value_concistency_loss = value_concistency_loss.item()

		self.optimizer_v.zero_grad(set_to_none=True)
		value_loss.backward()
		self.optimizer_v.step()

		# 2. Calculate Critic Loss
		with torch.no_grad():
			next_v = self.model_v(critic_next_observations)  # [batch_size, 1]
		target_q = critic_rewards + (1 - critic_dones) * self.gamma * next_v.detach()  # [batch_size, 1]
		
		if self.agent_model == 'illustrative':
			if self.continuous_actions:
				curr_q = self.model_qs(torch.concat([critic_observations, critic_actions], dim=-1))
			else:
				curr_q = self.model_qs(critic_observations).gather(-1, critic_actions.unsqueeze(0).repeat_interleave(self.value_ensemble_size, dim=0))
				assert qs.shape[-1] == 1
			#critic_loss = F.mse_loss(curr_q, target_q.unsqueeze(0).expand(self.value_ensemble_size, -1, -1)).mean()
			dims_to_mean_over = list(range(len(curr_q.shape)))[1:]
			critic_loss = ((curr_q - target_q.unsqueeze(0).expand(self.value_ensemble_size, -1, -1)) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)	

			critic_concistency_loss = 0
			if self.critic_da == "concistency":
				angles = [C4[random.randint(0, 3)] for _ in range(critic_observations.shape[0])]
				if self.continuous_actions:
					latent = self.model_qs.get_last_latent(torch.concat([critic_observations, critic_actions], dim=-1))
					augmented_latent = self.model_qs.get_last_latent(torch.concat([self.augmentation(critic_observations, angles), critic_actions], dim=-1))
				else:
					latent = self.model_qs.get_last_latent(critic_observations)
					augmented_latent = self.model_qs.get_last_latent(self.augmentation(critic_observations, angles))
				if self.detach_original:
					latent = latent.detach()
				dims_to_mean_over = list(range(len(latent.shape)))[1:]
				critic_concistency_loss = self.critic_concistency_coef * ((latent - augmented_latent) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)	
				critic_loss += critic_concistency_loss
				critic_concistency_loss = critic_concistency_loss.item()
			elif self.critic_da == "concistency_output":
				angles = [C4[random.randint(0, 3)] for _ in range(critic_observations.shape[0])]
				if self.continuous_actions:
					latent = self.model_qs(torch.concat([critic_observations, critic_actions], dim=-1))
					augmented_latent = self.model_qs(torch.concat([self.augmentation(critic_observations, angles), critic_actions], dim=-1))
				else:
					latent = self.model_qs(critic_observations)
					augmented_latent = self.model_qs(self.augmentation(critic_observations, angles))
				if self.detach_original:
					latent = latent.detach()
				dims_to_mean_over = list(range(len(latent.shape)))[1:]
				critic_concistency_loss = self.critic_concistency_coef * ((latent - augmented_latent) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)	
				critic_loss += critic_concistency_loss
				critic_concistency_loss = critic_concistency_loss.item()
			elif self.critic_da == "augment_concistency":
				angles = [C4[random.randint(0, 3)] for _ in range(observations.shape[0])]
				if self.continuous_actions:
					latent = self.model_qs(torch.concat([observations, actions], dim=-1))
					augmented_latent = self.model_qs(torch.concat([self.augmentation(observations, angles), actions], dim=-1))
				else:
					latent = self.model_qs(observations)
					augmented_latent = self.model_qs(self.augmentation(observations, angles))
				if self.detach_original:
					latent = latent.detach()
				dims_to_mean_over = list(range(len(latent.shape)))[1:]
				critic_concistency_loss = self.critic_concistency_coef * ((latent - augmented_latent) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)	
				critic_loss += critic_concistency_loss
				critic_concistency_loss = critic_concistency_loss.item()

			self.optimizer_qs.zero_grad(set_to_none=True)
			critic_loss.backward()
			self.optimizer_qs.step()
			avg_critic_loss = critic_loss.item() / self.value_ensemble_size
		else:
			avg_critic_loss = 0
			for i, m in enumerate(self.model_qs):
				if self.continuous_actions:
					curr_q = m(torch.concat([critic_observations, critic_actions], dim=-1))  # [batch_size, 1]
				else:
					curr_q = m(critic_observations).gather(1, critic_actions)  # [batch_size, 1]
				critic_loss = F.mse_loss(curr_q, target_q).mean()  # [1]

				critic_concistency_loss = 0
				if self.critic_da == "concistency":
					if self.continuous_actions:
						latent = m.get_last_latent(torch.concat([critic_observations, critic_actions], dim=-1))
						augmented_latent = m.get_last_latent(torch.concat([self.augmentation(critic_observations.clone()), critic_actions], dim=-1))
					else:
						latent = m.get_last_latent(critic_observations)
						augmented_latent = m.get_last_latent(self.augmentation(critic_observations.clone()))
					if self.detach_original:
						latent = latent.detach()
					critic_concistency_loss = self.critic_concistency_coef * F.mse_loss(latent, augmented_latent).mean()	
					critic_loss += critic_concistency_loss
					critic_concistency_loss = critic_concistency_loss.item()
				elif self.critic_da == "concistency_output":
					if self.continuous_actions:
						latent = m(torch.concat([critic_observations, critic_actions], dim=-1))
						augmented_latent = m(torch.concat([self.augmentation(critic_observations.clone()), critic_actions], dim=-1))
					else:
						latent = m(critic_observations)
						augmented_latent = m(self.augmentation(critic_observations.clone()))
					if self.detach_original:
						latent = latent.detach()
					critic_concistency_loss = self.critic_concistency_coef * F.mse_loss(latent, augmented_latent).mean()	
					critic_loss += critic_concistency_loss
					critic_concistency_loss = critic_concistency_loss.item()
				elif self.critic_da == "augment_concistency":
					if self.continuous_actions:
						latent = m(torch.concat([observations, actions], dim=-1))
						augmented_latent = m(torch.concat([self.augmentation(observations.clone()), actions], dim=-1))
					else:
						latent = m(observations)
						augmented_latent = m(self.augmentation(observations.clone()))
					if self.detach_original:
						latent = latent.detach()
					critic_concistency_loss = self.critic_concistency_coef * F.mse_loss(latent, augmented_latent).mean()	
					critic_loss += critic_concistency_loss
					critic_concistency_loss = critic_concistency_loss.item()

				self.optimizer_qs[i].zero_grad(set_to_none=True)
				critic_loss.backward()
				self.optimizer_qs[i].step()
				avg_critic_loss += critic_loss.item() / self.value_ensemble_size
		
		# Update the target network, copying all weights and biases in DQN
		if self.agent_model == 'illustrative':
			if self.perform_polyak_update:
				for target_param, param in zip(self.target_qs.parameters(), self.model_qs.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			else:
				if self.total_steps % self.target_update_freq == 0:
					self.target_qs.load_state_dict(self.model_qs.state_dict())
		else:
			for i in range(self.value_ensemble_size):
				if self.perform_polyak_update:
					for target_param, param in zip(self.target_qs[i].parameters(), self.model_qs[i].parameters()):
						target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
				else:
					if self.total_steps % self.target_update_freq == 0:
						self.target_qs[i].load_state_dict(self.model_qs[i].state_dict())

		# 3. Calculate Actor Loss
		if not self.continuous_actions and self.extract_all_actions:
			all_actions = torch.arange(self.action_space)
			actions = torch.repeat_interleave(all_actions, observations.shape[0]).to(observations.device).unsqueeze(-1)
			observations = observations.repeat(all_actions.shape[0], 1)
			
		if "augment" in self.actor_da:
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(observations.shape[0])]
				actor_observations_a = torch.concat([observations, self.augmentation(observations, angles)], dim=0)
			else:
				actor_observations_a = torch.concat([observations, self.augmentation(observations.clone())], dim=0)
			actor_actions_a = torch.concat([actions, actions], dim=0)
			if self.actor_da == 'augment_both':
				actor_observations_c = actor_observations_a
				actor_actions_c = actor_actions_a
			else:
				actor_observations_c = torch.concat([observations, observations], dim=0)
				actor_actions_c = actor_actions_a
		else:
			actor_observations_a = observations
			actor_actions_a = actions
			actor_observations_c = observations
			actor_actions_c = actions
		
		if self.use_value:
			with torch.no_grad():
				if self.agent_model == 'illustrative':
					if self.continuous_actions:
						qs = self.target_qs(torch.concat([actor_observations_c, actor_actions_c], dim=-1))
					else:
						qs = self.target_qs(actor_observations_c).gather(-1, actor_actions_c.unsqueeze(0).repeat_interleave(self.value_ensemble_size, dim=0))
						assert qs.shape[-1] == 1
				else:
					qs = []
					if self.continuous_actions:
						for t in self.target_qs:
							qs.append(t(torch.concat([actor_observations_c, actor_actions_c], dim=-1)))
					else:
						for t in self.target_qs:
							qs.append(t(actor_observations_c).gather(1, actor_actions_c))
					qs = torch.stack(qs, dim=0)		# [value_ensemble_size, batch_size, 1]
				if self.avg_q:
					q_avg = torch.mean(qs, dim=0) 	# [batch_size, 1]
				else:
					q_avg = torch.min(qs, dim=0)[0]
				curr_value = self.model_v(actor_observations_c)  # [batch_size, 1]

			actor_u_diff = q_avg - curr_value
		else:
			with torch.no_grad():
				if self.agent_model == 'illustrative':
					if self.continuous_actions:
						qs = self.target_qs(torch.concat([actor_observations_c, actor_actions_c], dim=-1))
					else:
						qs = self.target_qs(actor_observations_c).gather(-1, actor_actions_c.unsqueeze(0).repeat_interleave(self.value_ensemble_size, dim=0))
						assert qs.shape[-1] == 1
				else:
					qs = []
					if self.continuous_actions:
						for t in self.target_qs:
							qs.append(t(torch.concat([actor_observations_c, actor_actions_c], dim=-1)))
					else:
						for t in self.target_qs:
							qs.append(t(actor_observations_c).gather(1, actor_actions_c))
					qs = torch.stack(qs, dim=0)		# [value_ensemble_size, batch_size, 1]
				if self.avg_q:
					q_avg = torch.mean(qs, dim=0) 	# [batch_size, 1]
				else:
					q_avg = torch.min(qs, dim=0)[0]

			if self.continuous_actions:
				actor_u_diff = q_avg
			else:
				with torch.no_grad():
					if self.agent_model == 'illustrative':
						all_qs = self.target_qs(actor_observations_c)	# [value_ensemble_size, batch_size, n_actions]
						all_q_avg = torch.mean(all_qs, dim=0)
					else:
						all_qs = []
						for t in self.target_qs:
							all_qs.append(t(actor_observations_c))	# [value_ensemble_size, batch_size, n_actions]
						all_q_avg = torch.mean(torch.stack(all_qs, dim=0), dim=0)
				actor_u_diff = q_avg - torch.max(all_q_avg, dim=-1, keepdim=True)[0]
		exp_action = torch.exp(actor_u_diff.detach() * self.iql_temperature)  # [batch_size, 1]
		# take minimum of exp_action and 100.0 to avoid overflow
		exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
		# _, action_log_prob = self.get_action(observations, return_log_probs=True)  # [batch_size, 1]

		action_feats = self.model_actor(actor_observations_a)  # [actor_ensemble_size, batch_size, 2*action_dim] or [actor_ensemble_size, batch_size, n_actions]
		if not self.agent_model == 'illustrative':
			action_feats = action_feats.unsqueeze(0) 	# mimic ensemble of size 1
		
		#action_feats = torch.stack([m(actor_observations_a) for m in self.model_actor], dim=0)
		action_dist = self.actor_dist(action_feats)	# [actor_ensemble_size, batch_size] number of dists
		if self.continuous_actions:
			action_log_prob = action_dist.log_probs(self.normalise(actor_actions_a.unsqueeze(0).repeat_interleave(self.actor_ensemble_size, dim=0))) # [actor_ensemble_size, batch_size, action_dim] 
		else:
			action_log_prob = action_dist.log_probs(actor_actions_a.unsqueeze(0).repeat_interleave(self.actor_ensemble_size, dim=0)) # [actor_ensemble_size, batch_size, 1]
		dims_to_mean_over = list(range(len(action_log_prob.shape)))[1:]
		actor_loss = -((exp_action.broadcast_to(action_log_prob.shape) * action_log_prob).mean(dim=dims_to_mean_over).sum(dim=0))  # [1]


		actor_concistency_loss = 0
		if self.actor_da == "concistency":
			latent = self.model_actor.get_last_latent(actor_observations_a)
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(actor_observations_a.shape[0])]
				augmented_latent = self.model_actor.get_last_latent(self.augmentation(actor_observations_a, angles))
			else:
				augmented_latent = self.model_actor.get_last_latent(self.augmentation(actor_observations_a.clone()))
				latent = latent.unsqueeze(0) 	# mimic ensemble of size 1
				augmented_latent = augmented_latent.unsqueeze(0) 	# mimic ensemble of size 1
			if self.detach_original:
				latent = latent.detach()
			dims_to_mean_over = list(range(len(latent.shape)))[1:]
			actor_concistency_loss = self.actor_concistency_coef * ((latent - augmented_latent) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)
			actor_loss += actor_concistency_loss
			actor_concistency_loss = actor_concistency_loss.item()
		if self.actor_da == "augment_concistency":
			output = self.model_actor(observations)
			if self.consistency_probs:
				output = self.actor_dist(output)._get_probs()
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(observations.shape[0])]
				augmented_output = self.model_actor(self.augmentation(observations, angles))
				if self.consistency_probs:
					augmented_output = self.actor_dist(augmented_output)._get_probs()
			else:
				augmented_output = self.model_actor(self.augmentation(observations.clone()))
				output = output.unsqueeze(0) 	# mimic ensemble of size 1
				augmented_output = augmented_output.unsqueeze(0) 	# mimic ensemble of size 1
				if self.consistency_probs:
					augmented_output = self.actor_dist(augmented_output)._get_probs()
			if self.detach_original:
				output = output.detach()
			dims_to_mean_over = list(range(len(output.shape)))[1:]
			actor_concistency_loss = self.actor_concistency_coef * ((output - augmented_output) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)
			actor_loss += actor_concistency_loss
			actor_concistency_loss = actor_concistency_loss.item()
		elif self.actor_da == "concistency_output":
			output = self.model_actor(actor_observations_a)
			if self.consistency_probs:
				output = self.actor_dist(output)._get_probs()
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(actor_observations_a.shape[0])]
				augmented_output = self.model_actor(self.augmentation(actor_observations_a, angles))
				if self.consistency_probs:
					augmented_output = self.actor_dist(augmented_output)._get_probs()
			else:
				augmented_output = self.model_actor(self.augmentation(actor_observations_a.clone()))
				output = output.unsqueeze(0) 	# mimic ensemble of size 1
				augmented_output = augmented_output.unsqueeze(0) 	# mimic ensemble of size 1
				if self.consistency_probs:
					augmented_output = self.actor_dist(augmented_output)._get_probs()
			if self.detach_original:
				output = output.detach()
			dims_to_mean_over = list(range(len(output.shape)))[1:]
			actor_concistency_loss = self.actor_concistency_coef * ((output - augmented_output) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)
			actor_loss += actor_concistency_loss
			actor_concistency_loss = actor_concistency_loss.item()
		elif self.actor_da == "concistency_kl":
			output = self.actor_dist(self.model_actor(actor_observations_a))
			if self.agent_model == 'illustrative':
				angles = [C4[random.randint(0, 3)] for _ in range(actor_observations_a.shape[0])]
				augmented_output = self.actor_dist(self.model_actor(self.augmentation(actor_observations_a, angles)))
			else:
				augmented_output = self.actor_dist(self.model_actor(self.augmentation(actor_observations_a).clone()))
				output = output.unsqueeze(0) 	# mimic ensemble of size 1
				augmented_output = augmented_output.unsqueeze(0) 	# mimic ensemble of size 1
			if self.detach_original:
				output = output.detach()
			actor_concistency_loss = self.actor_concistency_coef * (torch.distributions.kl.kl_divergence(output, augmented_output).sum(dim=(1,2)) / actor_observations_a.shape[0]).sum(dim=0)
			actor_loss += actor_concistency_loss
			actor_concistency_loss = actor_concistency_loss.item()
		#elif self.actor_da == "concistency_soda":
		#	angles = [C4[random.randint(0, 3)] for _ in range(actor_observations_a.shape[0])]
		#	with torch.no_grad():
		#		output = self.soda_projector_target(self.target_actor(actor_observations_a))
		#	augmented_output = self.soda_predictor(self.soda_projector(self.model_actor(augmentation(actor_observations_a, angles))))
		#	output = F.normalize(output, dim=-1)
		#	augmented_output = F.normalize(augmented_output, dim=-1)
			#if not self.agent_model == 'illustrative':
			#	output = output.unsqueeze(0) 	# mimic ensemble of size 1
			#	augmented_output = augmented_output.unsqueeze(0) 	# mimic ensemble of size 1
		#	actor_concistency_loss = self.actor_concistency_coef * F.mse_loss(augmented_output, output).mean()	
		#	actor_loss += actor_concistency_loss
		#	actor_concistency_loss = actor_concistency_loss.item()

		self.optimizer_actor.zero_grad(set_to_none=True)
		#[o.zero_grad(set_to_none=True) for o in self.optimizer_actor]
		actor_loss.backward()
		self.optimizer_actor.step()
		#[o.step() for o in self.optimizer_actor]

		#if self.actor_da == "concistency_soda":
		#	# Update target networks
		#	for target_param, param in zip(self.target_actor.parameters(), self.model_actor.parameters()):
		#			target_param.data.copy_(self.actor_soda_update_coef * param.data + (1 - self.actor_soda_update_coef) * target_param.data)
		#	for target_param, param in zip(self.soda_projector_target.parameters(), self.soda_projector.parameters()):
		#		target_param.data.copy_(self.actor_soda_update_coef * param.data + (1 - self.actor_soda_update_coef) * target_param.data)

		self.total_steps += 1

		# create stats dict
		with torch.no_grad():
			loss = value_loss + actor_loss + avg_critic_loss
		stats = {
			"loss": loss.item(),
			"value_loss": value_loss.item(),
			"value_concistency_loss": value_concistency_loss,
			"avg_critic_loss": avg_critic_loss,
			"critic_concistency_loss": critic_concistency_loss,
			"actor_loss": actor_loss.item(),
			"actor_concistency_loss": actor_concistency_loss,
			"total_steps": self.total_steps,
		}
		# print(stats["actor_loss"])
		return stats
	
	#def train_actor(self, observations, actions, rewards, next_observations, dones):
	#	with torch.no_grad():
	#		qs = []
	#		if self.continuous_actions:
	#			for t in self.target_qs:
	#				qs.append(t(torch.concat([observations, actions], dim=-1)))
	#		else:
	#			for t in self.target_qs:
	#				qs.append(t(observations).gather(1, actions))
	#		qs = torch.stack(qs, dim=0)		# [ensembvalue_ensemble_sizele_size, batch_size, 1]
	#		q_avg = torch.mean(qs, dim=0) 	# [batch_size, 1]
	#		curr_value = self.model_v(observations)  # [batch_size, 1]

	#	if self.use_value:
	#		actor_u_diff = q_avg - curr_value
	#	else:
	#		if self.continuous_actions:
	#			actor_u_diff = q_avg
	#		else:
	#			with torch.no_grad():
	#				all_qs = []
	#				for t in self.target_qs:
	#					all_qs.append(t(observations))	# [value_ensemble_size, batch_size, n_actions]
	#				all_q_avg = torch.mean(torch.stack(all_qs, dim=0), dim=0)
	#			actor_u_diff = q_avg - torch.max(all_q_avg, dim=-1, keepdim=True)[0]
	#	exp_action = torch.exp(actor_u_diff.detach() * self.iql_temperature)  # [batch_size, 1]
	#	# take minimum of exp_action and 100.0 to avoid overflow
	#	exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
	#	# _, action_log_prob = self.get_action(observations, return_log_probs=True)  # [batch_size, 1]
	#	action_feats = self.model_actor(observations)  # [batch_size, 512]
	#	action_dist = self.actor_dist(action_feats)
	#	if self.continuous_actions:
	#		action_log_prob = action_dist.log_probs(self.normalise(actions))
	#	else:
	#		action_log_prob = action_dist.log_probs(actions)
	#	actor_loss = -(exp_action * action_log_prob).mean()  # [1]
	#	self.optimizer_actor.zero_grad(set_to_none=True)
	#	actor_loss.backward()
	#	self.optimizer_actor.step()

	#	self.total_steps += 1

	#	# create stats dict
	#	stats = {
	#		"actor_actor_loss": actor_loss.item(),
	#		"actor_total_steps": self.total_steps,
	#	}
	#	# print(stats["actor_loss"])
	#	return stats

	def eval_step(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0
		with torch.no_grad():
			if self.agent_model == 'illustrative':
				action_feats = self.model_actor(observations).mean(dim=0)
			else:
				action_feats = self.model_actor(observations)
			#action_feats = self.model_actor(observations)[0]
			#action_feats = self.model_actor[0](observations)
			action_dist = self.actor_dist(action_feats)

			if deterministic:
				action = action_dist.mode()
			else:
				action = action_dist.sample()  # [batch_size, 1]

			action_log_prob = action_dist.log_probs(action)

		if self.continuous_actions:
			action = self.unnormalise(action)

		if return_log_probs:
			return action.cpu().numpy(), action_log_prob

		return action.cpu().numpy()

	def get_action(self, observations, eps=0.0, return_log_probs=False):
		"""
		Given an observation, return an action.

		:param observation: the observation for the environment
		:param eps: the epsilon value for epsilon-greedy action selection
		:return: the action for the environment in numpy
		"""
		deterministic = eps == 0.0

		if self.agent_model == 'illustrative':
			action_feats = self.model_actor(observations).mean(dim=0)
		else:
			action_feats = self.model_actor(observations)
		#action_feats = self.model_actor(observations)[0]
		#action_feats = self.model_actor[0](observations)
		action_dist = self.actor_dist(action_feats)

		if deterministic:
			action = action_dist.mode()
		else:
			action = action_dist.sample()  # [batch_size, 1]

		action_log_prob = action_dist.log_probs(action)
		# st()
		# print(action_log_prob)

		if self.continuous_actions:
			action = self.unnormalise(action)

		if return_log_probs:
			return action, action_log_prob

		return action


	def save(self, num_epochs, path):
		"""
		Save the model to a given path.

		:param path: the path to save the model
		"""
		save_dict = {
			"actor_state_dict": self.model_actor.state_dict(),
			#"actor_dist_state_dict": self.actor_dist.state_dict(),
			"model_v_state_dict": self.model_v.state_dict(),
			"optimizer_actor_state_dict": self.optimizer_actor.state_dict(),
			"optimizer_v_state_dict": self.optimizer_v.state_dict(),
			"total_steps": self.total_steps,
			"curr_epochs": num_epochs,
		}
		if self.agent_model == 'illustrative':
			save_dict[f"model_q_state_dict"] = self.model_qs.state_dict()
			save_dict[f"target_q_state_dict"] = self.target_qs.state_dict()
			save_dict["optimizer_q_state_dict"] = self.optimizer_qs.state_dict()
		else:
			for i, m in enumerate(self.model_qs):
				save_dict[f"model_q{i}_state_dict"] = m.state_dict()
			for i, t in enumerate(self.target_qs):
				save_dict[f"target_q{i}_state_dict"] = t.state_dict()
			for i, o in enumerate(self.optimizer_qs):
				save_dict[f"optimizer_q{i}_state_dict"] = o.state_dict()

		#for i, m in enumerate(self.model_actor):
		#	save_dict[f"actor_{i}_state_dict"] = m.state_dict()
		#for i, o in enumerate(self.optimizer_actor):
		#	save_dict[f"optimizer_actor_{i}_state_dict"] = o.state_dict()

		torch.save(save_dict, path)
		return

	def load(self, path):
		"""
		Load the model from a given path.

		:param path: the path to load the model
		"""
		checkpoint = torch.load(path)
		self.model_actor.load_state_dict(checkpoint["actor_state_dict"])
		#self.actor_dist.load_state_dict(checkpoint["actor_dist_state_dict"])
		self.model_v.load_state_dict(checkpoint["model_v_state_dict"])
		self.optimizer_actor.load_state_dict(checkpoint["optimizer_actor_state_dict"])
		self.optimizer_v.load_state_dict(checkpoint["optimizer_v_state_dict"])
		self.total_steps = checkpoint["total_steps"]
		if self.agent_model == 'illustrative':
			self.model_qs.load_state_dict(checkpoint["model_q_state_dict"])
			self.target_qs.load_state_dict(checkpoint["target_q_state_dict"])
			self.optimizer_qs.load_state_dict(checkpoint["optimizer_q_state_dict"])
		else:
			for i, m in enumerate(self.model_qs):
				m.load_state_dict(checkpoint[f"model_q{i}_state_dict"])
			for i, t in enumerate(self.target_qs):
				t.load_state_dict(checkpoint[f"target_q{i}_state_dict"])
			for i, o in enumerate(self.optimizer_qs):
				o.load_state_dict(checkpoint[f"optimizer_q{i}_state_dict"])

		#for i, m in enumerate(self.model_actor):
		#	m.load_state_dict(checkpoint[f"actor_{i}_state_dict"])
		#for i, o in enumerate(self.optimizer_actor):
		#	o.load_state_dict(checkpoint[f"optimizer_actor_{i}_state_dict"])

		return checkpoint["curr_epochs"]