# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import time
import json
import csv
import numpy as np

import torch
import torch.nn as nn
from baselines.common.vec_env import VecExtractDictObs
from torch.utils.data import DataLoader

import wandb
from offline.agents import _create_agent
from offline.arguments import parser
from offline.dataloader import OfflineDataset, OfflineDTDataset
from offline.test_offline_agent import eval_agent, eval_DT_agent
from utils.filewriter import FileWriter
from utils.utils import set_seed
from utils.early_stopper import EarlyStop
import cProfile, pstats

from gym.envs.registration import register
import gym
register(
	 id="ControlIllustrativeCMDP-v0",
	 entry_point="control_illustrative_env:ControlIllustrativeCMDP",
)

args = parser.parse_args()
print(args)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

set_seed(args.seed)

if args.xpid is None:
	args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

# Setup wandb and offline logging
with open("wandb_info.txt") as file:
	lines = [line.rstrip() for line in file]
	# os.environ["WANDB_BASE_URL"] = lines[0]
	os.environ["WANDB_API_KEY"] = lines[1]
	os.environ["WANDB_START_METHOD"] = "thread"
	wandb_group = args.xpid[:-2][:126]  # '-'.join(args.xpid.split('-')[:-2])[:120]
	wandb_project = "Pessimism"
	wandb.init(project=wandb_project, entity=lines[2], config=args, name=args.xpid, group=wandb_group, tags=[args.algo, args.env_name, *args.wandb_tags])

log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
## check if final_model.pt already exists in the log_dir
#if os.path.exists(os.path.join(log_dir, args.xpid, "final_model.pt")):
#	# exit if final_model.pt already exists
#	print("Final model already exists in the log_dir")
#	exit(0)
filewriter = FileWriter(xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir)
filewriter.final_test_eval_fieldnames = ["final_test_ret", "final_train_ret", "final_val_ret", "final_total_variation"]
filewriter._finaltestwriter = csv.DictWriter(filewriter._finaltestfile, fieldnames=filewriter.final_test_eval_fieldnames)
filewriter._finaltestwriter.writeheader()


def log_stats(stats):
	filewriter.log(stats)
	wandb.log(stats)


# logging.getLogger().setLevel(logging.INFO)

#if args.env_name == "control_illustrative_eps0.5_symmetric" or args.env_name == "control_illustrative_eps0.5_non_symmetric":
if "data_symmetry" in args.dataset or "discrete" in args.dataset:
	set_id = int(args.xpid.split('_')[-1])
else:
	set_id = -1
#with open(f'datasets/task_sets_{set_id}.json', 'r') as file:
#	tasks_dict = json.load(file)
#with open(f'datasets/task_sets_base.json', 'r') as file:
#	tasks_dict = json.load(file)
env_name_split = args.env_name.split('_')
assert env_name_split[0] == 'control'
assert env_name_split[1] == 'illustrative'
env_type = '_'.join(args.env_name.split('_')[2:])
if set_id == -1:
	with open(f'{args.dataset}/task_sets_{env_type}.json', 'r') as file:
		tasks_dict = json.load(file)
else:
	with open(f'{args.dataset}/task_sets_{env_type}_{set_id}.json', 'r') as file:
		tasks_dict = json.load(file)

train_tasks = tasks_dict['base']
test_tasks = tasks_dict['test']

# Load dataset
pin_dataloader_memory = True
extra_config = None
if args.algo in ["dt", "bct"]:
	dataset = OfflineDTDataset(
		capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile, context_len=args.dt_context_length, rtg_noise_prob=args.dt_rtg_noise_prob
	)
	pin_dataloader_memory = True
	extra_config = {"train_data_vocab_size": dataset.vocab_size, "train_data_block_size": dataset._block_size, "max_timesteps": max(dataset._timesteps), "dataset_size": len(dataset)}
	eval_max_return = dataset.get_max_return(multiplier=args.dt_eval_ret)
	print("[DEBUG] Setting max eval return to ", eval_max_return)
else:
	if not set_id == -1:
		dataset = OfflineDataset(
			capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name + f'_{set_id}'), percentile=args.percentile
		)
	else:
		dataset = OfflineDataset(
			capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile
		)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_dataloader_memory) #, num_workers=8)

print("Dataset Loaded!")

## create Illustrative env
#env_kwargs = {'n_actions':3, 'simple_r_function':True}
#env_kwargs = {'n_actions':None, 'simple_r_function':False, 'epsilon': 0.02, 'terminal': False}
#env_kwargs = {'n_actions':None, 'simple_r_function':True, 'epsilon': 0.02, 'terminal': True}
if "discrete" in args.dataset or "value_distil" in args.dataset:
	env_kwargs = {'n_actions':3, 'simple_r_function':True, 'epsilon': 0.02, 'terminal': True}
else:
	env_kwargs = {'n_actions':None, 'simple_r_function':True, 'epsilon': 0.02, 'terminal': True}
env = gym.make('ControlIllustrativeCMDP-v0', tasks=train_tasks, **env_kwargs)

curr_epochs = 0
last_logged_update_count_at_restart = -1

# Initialize agent
agent = _create_agent(args, env=env, extra_config=extra_config)
agent.set_device(device)
print("Model Created!")

# wandb watch
# wandb.watch(agent.model_actor, log_freq=100)
# wandb.watch(agent.actor_dist, log_freq=100)
# wandb.watch(agent.model_v, log_freq=100)
# wandb.watch(agent.model_q1, log_freq=100)
# wandb.watch(agent.model_q2, log_freq=100)

# load checkpoint and resume if resume flag is true
if args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt")):
	curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
	last_logged_update_count_at_restart = filewriter.latest_update_count()
	print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")  
elif args.resume and os.path.exists(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt")):
	curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
	last_logged_update_count_at_restart = filewriter.latest_update_count()
	print(f"Resuming checkpoint from Epoch {curr_epochs}, logged update count {last_logged_update_count_at_restart}")
else:
	print("Starting from scratch!")

if args.early_stop:
	early_stopper = EarlyStop(wait_epochs=10, min_delta=0.1)

# Train agent
#with cProfile.Profile() as pr:
for epoch in range(curr_epochs, args.epochs):
	agent.train()
	epoch_loss = 0
	epoch_start_time = time.time()
	for observations, actions, rewards, next_observations, dones in dataloader:
		if len(actions.shape) == 1:
			actions = actions.unsqueeze(dim=1)
		if len(rewards.shape) == 1:
			rewards = rewards.unsqueeze(dim=1)
		if len(dones.shape) == 1:
			dones = dones.unsqueeze(dim=1)
		observations, actions, rewards, next_observations, dones = (
			observations.to(device),
			actions.to(device),
			rewards.to(device),
			next_observations.to(device),
			dones.to(device),
		)
		stats_dict = agent.train_step(
			observations.float(), actions, rewards.float(), next_observations.float(), dones.float()
		)
		epoch_loss += stats_dict["loss"]
	epoch_end_time = time.time()

	# evaluate the agent on illustrative environment
	if epoch % args.eval_freq == 0:
		inf_start_time = time.time()
		test_mean_perf, test_mean_len = eval_agent(
			agent,
			device,
			test_tasks,
			eval_eps=args.eval_eps,
			env_kwargs=env_kwargs
		)
		train_mean_perf, train_mean_len = eval_agent(
			agent,
			device,
			train_tasks,
			eval_eps=args.eval_eps,
			env_kwargs=env_kwargs
		)

		if "value_distil" in args.algo and args.policy_extraction == True:
			actor_one, actor_all, actor_sym, actor_non_sym = agent.eval_actors()
			test_perf_one, test_len_one = eval_agent(actor_one, device, test_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)
			train_perf_one, train_len_one = eval_agent(actor_one, device, train_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)

			test_perf_all, test_len_all = eval_agent(actor_all, device, test_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)
			train_perf_all, train_len_all = eval_agent(actor_all, device, train_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)

			test_perf_sym, test_len_sym = eval_agent(actor_sym, device, test_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)
			train_perf_sym, train_len_sym = eval_agent(actor_sym, device, train_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)

			test_perf_non_sym, test_len_non_sym = eval_agent(actor_non_sym, device, test_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)
			train_perf_non_sym, train_len_non_sym = eval_agent(actor_non_sym, device, train_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)

		inf_end_time = time.time()

		print(
			f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(dataloader)} | Time: {epoch_end_time - epoch_start_time} \
				| Train Returns (mean): {train_mean_perf} | Test Returns (mean): {test_mean_perf}"
		)

		print(epoch+1)
		if (epoch+1) > last_logged_update_count_at_restart:
			stats_dict.update(
				{
					"epoch": epoch + 1,
					"train_loss": epoch_loss / len(dataloader),
					"epoch_time": epoch_end_time - epoch_start_time,
					"inf_time": inf_end_time - inf_start_time,
					"train_rets_mean": train_mean_perf,
					"train_len_mean": train_mean_len,
					"test_rets_mean": test_mean_perf,
					"test_len_mean": test_mean_len
				}
			)
			if "value_distil" in args.algo and args.policy_extraction == True:
				stats_dict.update(
					{
						"policy_extraction/train_rets_mean_one": train_perf_one,
						"policy_extraction/train_len_mean_one": train_len_one,
						"policy_extraction/test_rets_mean_one": test_perf_one,
						"policy_extraction/test_len_mean_one": test_len_one,

						"policy_extraction/train_rets_mean_all": train_perf_all,
						"policy_extraction/train_len_mean_all": train_len_all,
						"policy_extraction/test_rets_mean_all": test_perf_all,
						"policy_extraction/test_len_mean_all": test_len_all,

						"policy_extraction/train_rets_mean_sym": train_perf_sym,
						"policy_extraction/train_len_mean_sym": train_len_sym,
						"policy_extraction/test_rets_mean_sym": test_perf_sym,
						"policy_extraction/test_len_mean_sym": test_len_sym,

						"policy_extraction/train_rets_mean_non_sym": train_perf_non_sym,
						"policy_extraction/train_len_mean_non_sym": train_len_non_sym,
						"policy_extraction/test_rets_mean_non_sym": test_perf_non_sym,
						"policy_extraction/test_len_mean_non_sym": test_len_non_sym,
					}
				)
			log_stats(stats_dict)

	# Save agent and number of epochs
	if args.resume and (epoch+1) % args.ckpt_freq == 0:
		curr_epochs = epoch + 1
		agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
		agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, f"model_{epoch}.pt"))
				
test_mean_perf, test_mean_len = eval_agent(agent, device, test_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)
train_mean_perf, train_mean_len = eval_agent(agent, device, train_tasks, eval_eps=args.eval_eps, env_kwargs=env_kwargs)


	#if 'iql' in args.algo:
	#	# Train the final policy
	#	print(f"Extracting the Actor after training")
	#	agent.reset_actor()
	#	for epoch in range(0, 30):
	#		agent.train()
	#		epoch_loss = 0
	#		epoch_start_time = time.time()
	#		for observations, actions, rewards, next_observations, dones in dataloader:
	#			if len(actions.shape) == 1:
	#				actions = actions.unsqueeze(dim=1)
	#			if len(rewards.shape) == 1:
	#				rewards = rewards.unsqueeze(dim=1)
	#			if len(dones.shape) == 1:
	#				dones = dones.unsqueeze(dim=1)
	#			observations, actions, rewards, next_observations, dones = (
	#				observations.to(device),
	#				actions.to(device),
	#				rewards.to(device),
	#				next_observations.to(device),
	#				dones.to(device),
	#			)
	#			stats_dict = agent.train_actor(
	#				observations.float(), actions, rewards.float(), next_observations.float(), dones.float()
	#			)
	#			epoch_loss += stats_dict["actor_actor_loss"]
	#		epoch_end_time = time.time()

	#		# evaluate the agent on illustrative environment
	#		if epoch % 1 == 0:
	#			inf_start_time = time.time()
	#			test_mean_perf, test_mean_len = eval_agent(
	#				agent,
	#				device,
	#				test_tasks,
	#				eval_eps=args.eval_eps,
	#				env_kwargs=env_kwargs
	#			)
	#			train_mean_perf, train_mean_len = eval_agent(
	#				agent,
	#				device,
	#				train_tasks,
	#				eval_eps=args.eval_eps,
	#				env_kwargs=env_kwargs
	#			)
	#			inf_end_time = time.time()

	#			print(
	#				f"Epoch: {epoch + 1} | Loss: {epoch_loss / len(dataloader)} | Time: {epoch_end_time - epoch_start_time} \
	#					| Train Returns (mean): {train_mean_perf} | Test Returns (mean): {test_mean_perf}"
	#			)

	#			print(epoch+1)
	#			stats_dict.update(
	#				{
	#					"actor_epoch": epoch + 1,
	#					"actor_train_loss": epoch_loss / len(dataloader),
	#					#"epoch_time": epoch_end_time - epoch_start_time,
	#					#"inf_time": inf_end_time - inf_start_time,
	#					"actor_train_rets_mean": train_mean_perf,
	#					"actor_train_len_mean": train_mean_len,
	#					"actor_test_rets_mean": test_mean_perf,
	#					"actor_test_len_mean": test_mean_len
	#				}
	#			)
	#			wandb.log(stats_dict)

#stats = pstats.Stats(pr)
#stats.sort_stats(pstats.SortKey.TIME)
#stats.dump_stats(filename=f"profiling.prof")

if "iql" in args.algo:
	total_variation = []
	total_variation_q = []
	total_variation_v = []
	obs = []
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=test_tasks, **env_kwargs)
	for _ in range(len(test_tasks)):
		obs.append(env.reset())
	obs = np.array(obs)
	obs = torch.as_tensor(obs, device=device)
	with torch.no_grad():
		#output = agent.model_base(obs).mean(dim=0)
		output = agent.model_actor(obs)
		if len(output.shape) == 3:
			# average over ensemble
			output = output.mean(dim=0)

		if env_kwargs["n_actions"] == None:
			actions = agent.unnormalise(output[:, :2])
			qs = agent.target_qs(torch.concat([obs, actions], dim=-1)) # [iql_value_ensemble_size, batch_size, 1]
		else:
			qs = agent.target_qs(obs)
		if args.iql_avg_q:
			qs = torch.mean(qs, dim=0) 	# [batch_size, 1]
		else:
			qs = torch.min(qs, dim=0)[0]	# [batch_size, 1]

		v = agent.model_v(obs)	# [batch_size, 1]

		

	total_variation.append(np.trace(np.cov(output.cpu().numpy(), rowvar=False)))
	total_variation = np.mean(total_variation)

	if env_kwargs["n_actions"] == None:
		total_variation_q.append(np.var(qs.cpu().numpy()))
	else:
		total_variation_q.append(np.trace(np.cov(qs.cpu().numpy(), rowvar=False)))
	total_variation_q = np.mean(total_variation_q)

	total_variation_v.append(np.var(v.cpu().numpy()))
	total_variation_v = np.mean(total_variation_v)


	wandb.log({"final_total_variation_q": total_variation_q, "final_total_variation_v": total_variation_v, "final_total_variation": total_variation, "final_test_ret": test_mean_perf, "final_test_len": test_mean_len, "final_train_ret": train_mean_perf, "final_train_len": train_mean_len}, step=(epoch + 1))

	filewriter.log_final_test_eval({
			'final_test_ret': test_mean_perf,
			'final_train_ret': train_mean_perf,
			'final_total_variation': total_variation
		})
elif "value_distil" in args.algo:
	total_variation = []
	if agent.policy_extraction:
		total_variation_one_action = []
		total_variation_all_actions = []
		total_variation_sym_actions = []
		total_variation_non_sym_actions = []
	obs = []
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=test_tasks, **env_kwargs)
	for _ in range(len(test_tasks)):
		obs.append(env.reset())
	obs = np.array(obs)
	obs = torch.as_tensor(obs, device=device)
	with torch.no_grad():
		output = agent.model_base(obs).mean(dim=0)
		if agent.policy_extraction:
			out_one = agent.actor_one_action(obs).mean(dim=0)
			out_all = agent.actor_all_actions(obs).mean(dim=0)
			out_sym = agent.actor_sym_actions(obs).mean(dim=0)
			out_non_sym = agent.actor_non_sym_actions(obs).mean(dim=0)
			

	total_variation.append(np.trace(np.cov(output.cpu().numpy(), rowvar=False)))
	total_variation = np.mean(total_variation)
	if agent.policy_extraction:
		total_variation_one_action.append(np.trace(np.cov(out_one.cpu().numpy(), rowvar=False)))
		total_variation_one_action = np.mean(total_variation_one_action)
		total_variation_all_actions.append(np.trace(np.cov(out_all.cpu().numpy(), rowvar=False)))
		total_variation_all_actions = np.mean(total_variation_all_actions)
		total_variation_sym_actions.append(np.trace(np.cov(out_sym.cpu().numpy(), rowvar=False)))
		total_variation_sym_actions = np.mean(total_variation_sym_actions)
		total_variation_non_sym_actions.append(np.trace(np.cov(out_non_sym.cpu().numpy(), rowvar=False)))
		total_variation_non_sym_actions = np.mean(total_variation_non_sym_actions)
		wandb.log({"policy_extraction/final_total_variation_non_sym_actions": total_variation_non_sym_actions, "policy_extraction/final_total_variation_sym_actions": total_variation_sym_actions, "policy_extraction/final_total_variation_all_actions": total_variation_all_actions, "policy_extraction/final_total_variation_one_action": total_variation_one_action, "final_total_variation": total_variation, "final_test_ret": test_mean_perf, "final_test_len": test_mean_len, "final_train_ret": train_mean_perf, "final_train_len": train_mean_len}, step=(epoch + 1))

	else:
		wandb.log({"final_total_variation": total_variation, "final_test_ret": test_mean_perf, "final_test_len": test_mean_len, "final_train_ret": train_mean_perf, "final_train_len": train_mean_len}, step=(epoch + 1))

	filewriter.log_final_test_eval({
			'final_test_ret': test_mean_perf,
			'final_train_ret': train_mean_perf,
			'final_total_variation': total_variation
		})
elif "cql" in args.algo:
	total_variation = []
	obs = []
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=test_tasks, **env_kwargs)
	for _ in range(len(test_tasks)):
		obs.append(env.reset())
	obs = np.array(obs)
	obs = torch.as_tensor(obs, device=device)
	with torch.no_grad():
		output = agent.model(obs)

	total_variation.append(np.trace(np.cov(output.cpu().numpy(), rowvar=False)))
	total_variation = np.mean(total_variation)

	wandb.log({"final_total_variation": total_variation, "final_test_ret": test_mean_perf, "final_test_len": test_mean_len, "final_train_ret": train_mean_perf, "final_train_len": train_mean_len}, step=(epoch + 1))

	filewriter.log_final_test_eval({
			'final_test_ret': test_mean_perf,
			'final_train_ret': train_mean_perf,
			'final_total_variation': total_variation
		})
	

#if args.resume:
agent.save(num_epochs=args.epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
