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
import copy
import random


class BehavioralCloningEnsemble:
	def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64, ensemble_size=1, **kwargs):
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
		self.ensemble_size = ensemble_size

		self.model_base = AGENT_CLASSES[agent_model](observation_space, action_space, hidden_size, use_actor_linear=True, ensemble_size=ensemble_size, **kwargs)
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
			# [ensemble_size, batch_size, out_dim]
			actor_features = self.model_base(observation)
			dists = [FixedCategorical(logits=af) for af in actor_features]

			# # take the mean over the logits
			# logits = torch.stack([d._get_logits() for d in dists]).mean(dim=0)
			# dist = FixedCategorical(logits=logits)
			# take the mean over the probs
			probs = torch.stack([d.probs for d in dists]).mean(dim=0)
			dist = FixedCategorical(probs=probs)
			
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
		actions = actions.long()

		# squeeze actions to [batch_size] if they are [batch_size, 1]
		if len(actions.shape) == 2:
			actions = actions.squeeze(dim=1)
			
		# [ensemble_size, batch_size, out_dim]
		actor_features = self.model_base(observations)
		dists = [FixedCategorical(logits=af) for af in actor_features]
		action_log_probs = torch.cat([d._get_log_softmax() for d in dists], dim=0)
		
		self.optimizer.zero_grad()
		loss = F.nll_loss(action_log_probs, actions.repeat(self.ensemble_size))
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
	


class BehavioralCloningEnsembleContinuous:
	def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64, ensemble_size=1, **kwargs):
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
		self.ensemble_size = ensemble_size
		self.low = torch.as_tensor(action_space.low).float()
		self.high = torch.as_tensor(action_space.high).float()

		self.model_base = AGENT_CLASSES[agent_model](observation_space, action_space.shape[0], hidden_size, use_actor_linear=True, ensemble_size=ensemble_size, **kwargs)
		self.optimizer = torch.optim.Adam(self.model_base.parameters(), lr=self.lr)
		
		self.total_steps = 0

	def train(self):
		self.model_base.train()

	def eval(self):
		self.model_base.eval()

	def set_device(self, device):
		self.model_base.to(device)
		self.low = self.low.to(device)
		self.high = self.high.to(device)

	def unnormalise(self, x):
		# turn x from range [-1, 1] to [self.low, self.high]
		x = torch.tanh(x)
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
			# [ensemble_size, batch_size, out_dim]
			unbound_output = self.model_base(observation)
			action = self.unnormalise(unbound_output).squeeze(-1)
			action = action.mean(dim=0)


		return action.cpu().numpy()

	def train_step(self, observations, actions, rewards, next_observations, dones):
		"""
		Update the agent given observations and actions.

		:param observations: the observations for the environment
		:param actions: the actions for the environment
		"""
		actions = actions.float()

		# squeeze actions to [batch_size] if they are [batch_size, 1]
		if len(actions.shape) == 2:
			actions = actions.squeeze(dim=1)
			
		# [ensemble_size, batch_size, out_dim]
		unbound_output = self.model_base(observations)
		policy_output = self.unnormalise(unbound_output).squeeze(-1)

		
		self.optimizer.zero_grad()
		dims_to_mean_over = list(range(len(policy_output.shape)))[1:]
		loss = ((policy_output - actions.broadcast_to(policy_output.shape).float()) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)
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
	

class ValueDistilEnsemble:
	def __init__(self, observation_space, action_space, lr, agent_model, hidden_size=64, ensemble_size=1, policy_extraction=False, policy_extraction_temp=3, **kwargs):
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
		self.ensemble_size = ensemble_size
		self.policy_extraction = policy_extraction
		self.policy_extraction_temp = policy_extraction_temp

		self.model_base = AGENT_CLASSES[agent_model](observation_space, action_space.n, hidden_size, use_actor_linear=True, ensemble_size=ensemble_size, **kwargs)
		self.optimizer = torch.optim.Adam(self.model_base.parameters(), lr=self.lr)
		
		if self.policy_extraction:
			self.actor_dist = Categorical(hidden_size, self.action_space.n)
			# Actor that is cloned on only the optimal action
			self.actor_one_action = AGENT_CLASSES[agent_model](
				observation_space, action_space.n, hidden_size, use_actor_linear=True, ensemble_size=1, **kwargs,
			)
			self.optimizer_one_action = torch.optim.Adam(self.actor_one_action.parameters(), lr=self.lr)
			# Actor that is cloned on all actions
			self.actor_all_actions = AGENT_CLASSES[agent_model](
				observation_space, action_space.n, hidden_size, use_actor_linear=True, ensemble_size=1, **kwargs,
			)
			self.optimizer_all_actions = torch.optim.Adam(self.actor_all_actions.parameters(), lr=self.lr)
			# Actor that is cloned on a random subset of actions (plus the optimal one), symmetric across the contexts
			# suboptimal actions (that appear to be suboptimal on all training tasks): [0,1,3,4,6]
			self.actor_sym_actions = AGENT_CLASSES[agent_model](
				observation_space, action_space.n, hidden_size, use_actor_linear=True, ensemble_size=1, **kwargs,
			)
			self.optimizer_sym_actions = torch.optim.Adam(self.actor_sym_actions.parameters(), lr=self.lr)
			self.sym_actions = set(np.random.choice(self.action_space.n, 4, replace=False))
			# Actor that is cloned on a random subset of actions (plus the optimal one), non-symmetric across the contexts
			self.actor_non_sym_actions = AGENT_CLASSES[agent_model](
				observation_space, action_space.n, hidden_size, use_actor_linear=True, ensemble_size=1, **kwargs,
			)
			self.optimizer_non_sym_actions = torch.optim.Adam(self.actor_non_sym_actions.parameters(), lr=self.lr)
			self.non_sym_actions_1 = set(np.random.choice(self.action_space.n, 4, replace=False))
			self.non_sym_actions_2 = set(np.arange(action_space.n)) - self.non_sym_actions_1
			non_sym_subset1 = set(random.sample(self.non_sym_actions_1, 2))
			non_sym_subset2 = set(random.sample(self.non_sym_actions_2, 3))
			self.non_sym_actions_3 = non_sym_subset1 | non_sym_subset2
			self.non_sym_actions_4 = (self.non_sym_actions_1 - non_sym_subset1) | (self.non_sym_actions_2 - non_sym_subset2)
			
		
		self.total_steps = 0

	def train(self):
		self.model_base.train()
		if self.policy_extraction:
			self.actor_one_action.train()
			self.actor_all_actions.train()
			self.actor_sym_actions.train()
			self.actor_non_sym_actions.train()

	def eval(self):
		self.model_base.eval()
		if self.policy_extraction:
			self.actor_one_action.eval()
			self.actor_all_actions.eval()
			self.actor_sym_actions.eval()
			self.actor_non_sym_actions.eval()

	def set_device(self, device):
		self.model_base.to(device)
		if self.policy_extraction:
			self.actor_one_action.to(device)
			self.actor_all_actions.to(device)
			self.actor_sym_actions.to(device)
			self.actor_non_sym_actions.to(device)

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
			# [ensemble_size, batch_size, out_dim]
			values = self.model_base(observation).mean(dim=0)
			action = values.argmax(dim=-1)


		return action.cpu().numpy()
	
	def eval_actors(self):
		"""
		Given an observation, return an action for each of the actors.

		:param observation: the observation for the environment
		:return: the action for the environment in numpy
		"""
		class EvalActor(nn.Module):
			def __init__(self, model, dist):
				super().__init__()
				self.model = model
				self.dist = dist

			def eval(self):
				self.model.eval()
			def eval_step(self, observation, eps=0.0):
				with torch.no_grad():
					action = self.dist(self.model(observation).mean(dim=0)).mode()
				return action.cpu().numpy()
			
		if self.policy_extraction:
			return EvalActor(self.actor_one_action, self.actor_dist), EvalActor(self.actor_all_actions, self.actor_dist), EvalActor(self.actor_sym_actions, self.actor_dist), EvalActor(self.actor_non_sym_actions, self.actor_dist)

	def train_step(self, observations, values, rewards, next_observations, dones):
		"""
		Update the agent given observations and actions.

		:param observations: the observations for the environment
		:param actions: the actions for the environment
		"""
		values = values.float()

		if len(values.shape) == 2:
			values = values.squeeze(dim=1)
			
		# [ensemble_size, batch_size, out_dim]
		output = self.model_base(observations)

		
		self.optimizer.zero_grad()
		dims_to_mean_over = list(range(len(output.shape)))[1:]
		loss = ((output - values.broadcast_to(output.shape).float()) ** 2).mean(dim=dims_to_mean_over).sum(dim=0)
		loss.backward()
		self.optimizer.step()

		# create stats dict
		stats = {"loss": loss.item(), "total_steps": self.total_steps}


		if self.policy_extraction:
			# Weighted BC loss for only optimal action
			optimal_action = torch.argmax(values, dim=-1).unsqueeze(-1) 	# [batch_size, 1]
			u_diff = values.gather(1, optimal_action) - torch.max(values, dim=-1)[0].unsqueeze(-1)
			exp_action = torch.exp(u_diff.detach() * self.policy_extraction_temp)
			# take minimum of exp_action and 100.0 to avoid overflow
			exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
			action_dist = self.actor_dist(self.actor_one_action(observations).mean(dim=0))
			action_log_prob = action_dist.log_probs(optimal_action)
			actor_loss_one = -(exp_action * action_log_prob).mean()
			self.optimizer_one_action.zero_grad(set_to_none=True)
			actor_loss_one.backward()
			self.optimizer_one_action.step()

			# Weighted BC loss for all actions
			all_actions = torch.arange(self.action_space.n)
			repeated_actions = torch.repeat_interleave(all_actions, observations.shape[0]).to(observations.device).unsqueeze(-1)
			repeated_observations = observations.repeat(all_actions.shape[0], 1)
			repeated_values = values.repeat(all_actions.shape[0], 1)
			u_diff = repeated_values.gather(1, repeated_actions) - torch.max(repeated_values, dim=-1)[0].unsqueeze(-1)
			exp_action = torch.exp(u_diff.detach() * self.policy_extraction_temp)
			# take minimum of exp_action and 100.0 to avoid overflow
			exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
			action_dist = self.actor_dist(self.actor_all_actions(repeated_observations).mean(dim=0))
			action_log_prob = action_dist.log_probs(repeated_actions)
			actor_loss_all = -(exp_action * action_log_prob).mean()
			self.optimizer_all_actions.zero_grad(set_to_none=True)
			actor_loss_all.backward()
			self.optimizer_all_actions.step()

			# Weighted BC loss for random symmetric actions
			sym_actions = torch.as_tensor(list(self.sym_actions), dtype=torch.int64, device=observations.device)
			repeated_actions = torch.repeat_interleave(sym_actions, observations.shape[0]).to(observations.device).unsqueeze(-1)
			repeated_actions = torch.concat([optimal_action, repeated_actions], dim=0)
			repeated_observations = observations.repeat(sym_actions.shape[0], 1)
			repeated_observations = torch.concat([observations, repeated_observations], dim=0)
			repeated_values = values.repeat(sym_actions.shape[0], 1)
			repeated_values = torch.concat([values, repeated_values], dim=0)
			u_diff = repeated_values.gather(1, repeated_actions) - torch.max(repeated_values, dim=-1)[0].unsqueeze(-1)
			exp_action = torch.exp(u_diff.detach() * self.policy_extraction_temp)
			# take minimum of exp_action and 100.0 to avoid overflow
			exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
			action_dist = self.actor_dist(self.actor_sym_actions(repeated_observations).mean(dim=0))
			action_log_prob = action_dist.log_probs(repeated_actions)
			actor_loss_sym = -(exp_action * action_log_prob).mean()
			self.optimizer_sym_actions.zero_grad(set_to_none=True)
			actor_loss_sym.backward()
			self.optimizer_sym_actions.step()

			# Weighted BC loss for random non-symmetric actions
			repeated_actions = optimal_action
			repeated_observations = observations
			repeated_values = values

			# Context 1
			context_1_indices = (torch.round(observations[:, :2]) == torch.tensor([0, 1], device=observations.device).unsqueeze(0)).all(dim=-1)
			non_sym_actions_1 = torch.as_tensor(list(self.non_sym_actions_1), dtype=torch.int64, device=observations.device)
			repeated_actions_1 = torch.repeat_interleave(non_sym_actions_1, observations[context_1_indices].shape[0]).to(observations.device).unsqueeze(-1)
			repeated_actions = torch.concat([repeated_actions, repeated_actions_1], dim=0)
			repeated_observations_1 = observations[context_1_indices].repeat(non_sym_actions_1.shape[0], 1)
			repeated_observations = torch.concat([repeated_observations, repeated_observations_1], dim=0)
			repeated_values_1 = values[context_1_indices].repeat(non_sym_actions_1.shape[0], 1)
			repeated_values = torch.concat([repeated_values, repeated_values_1], dim=0)

			# Context 2
			context_2_indices = (torch.round(observations[:, :2]) == torch.tensor([0, -1], device=observations.device).unsqueeze(0)).all(dim=-1)
			non_sym_actions_2 = torch.as_tensor(list(self.non_sym_actions_2), dtype=torch.int64, device=observations.device)
			repeated_actions_2 = torch.repeat_interleave(non_sym_actions_2, observations[context_2_indices].shape[0]).to(observations.device).unsqueeze(-1)
			repeated_actions = torch.concat([repeated_actions, repeated_actions_2], dim=0)
			repeated_observations_2 = observations[context_2_indices].repeat(non_sym_actions_2.shape[0], 1)
			repeated_observations = torch.concat([repeated_observations, repeated_observations_2], dim=0)
			repeated_values_2 = values[context_2_indices].repeat(non_sym_actions_2.shape[0], 1)
			repeated_values = torch.concat([repeated_values, repeated_values_2], dim=0)

			# Context 3
			context_3_indices = (torch.round(observations[:, :2]) == torch.tensor([1, 0], device=observations.device).unsqueeze(0)).all(dim=-1)
			non_sym_actions_3 = torch.as_tensor(list(self.non_sym_actions_3), dtype=torch.int64, device=observations.device)
			repeated_actions_3 = torch.repeat_interleave(non_sym_actions_3, observations[context_3_indices].shape[0]).to(observations.device).unsqueeze(-1)
			repeated_actions = torch.concat([repeated_actions, repeated_actions_3], dim=0)
			repeated_observations_3 = observations[context_3_indices].repeat(non_sym_actions_3.shape[0], 1)
			repeated_observations = torch.concat([repeated_observations, repeated_observations_3], dim=0)
			repeated_values_3 = values[context_3_indices].repeat(non_sym_actions_3.shape[0], 1)
			repeated_values = torch.concat([repeated_values, repeated_values_3], dim=0)

			# Context 4
			context_4_indices = (torch.round(observations[:, :2]) == torch.tensor([-1, 0], device=observations.device).unsqueeze(0)).all(dim=-1)
			non_sym_actions_4 = torch.as_tensor(list(self.non_sym_actions_4), dtype=torch.int64, device=observations.device)
			repeated_actions_4 = torch.repeat_interleave(non_sym_actions_4, observations[context_4_indices].shape[0]).to(observations.device).unsqueeze(-1)
			repeated_actions = torch.concat([repeated_actions, repeated_actions_4], dim=0)
			repeated_observations_4 = observations[context_4_indices].repeat(non_sym_actions_4.shape[0], 1)
			repeated_observations = torch.concat([repeated_observations, repeated_observations_4], dim=0)
			repeated_values_4 = values[context_4_indices].repeat(non_sym_actions_4.shape[0], 1)
			repeated_values = torch.concat([repeated_values, repeated_values_4], dim=0)

			u_diff = repeated_values.gather(1, repeated_actions) - torch.max(repeated_values, dim=-1)[0].unsqueeze(-1)
			exp_action = torch.exp(u_diff.detach() * self.policy_extraction_temp)
			# take minimum of exp_action and 100.0 to avoid overflow
			exp_action = torch.min(exp_action, torch.tensor(100.0).to(exp_action.device))  # [batch_size, 1]
			action_dist = self.actor_dist(self.actor_non_sym_actions(repeated_observations).mean(dim=0))
			action_log_prob = action_dist.log_probs(repeated_actions)
			actor_loss_non_sym = -(exp_action * action_log_prob).mean()
			self.optimizer_non_sym_actions.zero_grad(set_to_none=True)
			actor_loss_non_sym.backward()
			self.optimizer_non_sym_actions.step()

			stats["actor_loss_one"] = actor_loss_one.item()
			stats["actor_loss_all"] = actor_loss_all.item()
			stats["actor_loss_sym"] = actor_loss_sym.item()
			stats["actor_loss_non_sym"] = actor_loss_non_sym.item()

		self.total_steps += 1

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
		if self.policy_extraction:
			save_dict["actor_one"] = self.actor_one_action.state_dict()
			save_dict["optimizer_actor_one"] = self.optimizer_one_action.state_dict()
			save_dict["actor_all"] = self.actor_all_actions.state_dict()
			save_dict["optimizer_actor_all"] = self.optimizer_all_actions.state_dict()
			save_dict["actor_sym"] = self.actor_sym_actions.state_dict()
			save_dict["optimizer_actor_sym"] = self.optimizer_sym_actions.state_dict()
			save_dict["actor_non_sym"] = self.actor_non_sym_actions.state_dict()
			save_dict["optimizer_actor_non_sym"] = self.optimizer_non_sym_actions.state_dict()

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
		if self.policy_extraction:
			self.actor_one_action.load_state_dict(checkpoint["actor_one"])
			self.optimizer_one_action.load_state_dict(checkpoint["optimizer_actor_one"])
			self.actor_all_actions.load_state_dict(checkpoint["actor_all"])
			self.optimizer_all_actions.load_state_dict(checkpoint["optimizer_actor_all"])
			self.actor_sym_actions.load_state_dict(checkpoint["actor_sym"])
			self.optimizer_sym_actions.load_state_dict(checkpoint["optimizer_actor_sym"])
			self.actor_non_sym_actions.load_state_dict(checkpoint["actor_non_sym"])
			self.optimizer_non_sym_actions.load_state_dict(checkpoint["optimizer_actor_non_sym"])

		return checkpoint["curr_epochs"]
	

