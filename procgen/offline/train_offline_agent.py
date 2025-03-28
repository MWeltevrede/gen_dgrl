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

from gym.envs.registration import register
import gym
register(
     id="GridIllustrativeCMDPContinuous-v0",
     entry_point="grid_illustrative_env:IllustrativeCMDPContinuous",
)
register(
     id="GridIllustrativeCMDPDiscrete-v0",
     entry_point="grid_illustrative_env:IllustrativeCMDPDiscrete",
)
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
    wandb_project = "OfflineRLBenchmark"
    wandb.init(project=wandb_project, entity=lines[2], config=args, name=args.xpid, group=wandb_group, tags=[args.algo, args.env_name])

log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
# check if final_model.pt already exists in the log_dir
if os.path.exists(os.path.join(log_dir, args.xpid, "final_model.pt")):
    # exit if final_model.pt already exists
    print("Final model already exists in the log_dir")
    exit(0)
filewriter = FileWriter(xpid=args.xpid, xp_args=args.__dict__, rootdir=log_dir)


def log_stats(stats):
    filewriter.log(stats)
    wandb.log(stats)


# logging.getLogger().setLevel(logging.INFO)

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
    dataset = OfflineDataset(
        capacity=args.dataset_size, episodes_dir_path=os.path.join(args.dataset, args.env_name), percentile=args.percentile
    )
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=pin_dataloader_memory) #, num_workers=8)

print("Dataset Loaded!")

# create Illustrative env
if args.env_name == "grid_illustrative":
    # Alternating subgroup
    train_tasks = [((255,0,128), 'left'), ((255,0,128), 'right'), ((255,0,128), 'top'), ((255,0,128), 'bottom'),
                ((0,255,128), 'left'), ((0,255,128), 'right'), ((0,255,128), 'top'), ((0,255,128), 'bottom')]
    # Full group of permutations
    test_tasks = [((128,255,0), 'left'), ((128,255,0), 'right'), ((128,255,0), 'top'), ((128,255,0), 'bottom'),
                ((255,128,0), 'left'), ((255,128,0), 'right'), ((255,128,0), 'top'), ((255,128,0), 'bottom'),
                ((128,0,255), 'left'), ((128,0,255), 'right'), ((128,0,255), 'top'), ((128,0,255), 'bottom'),
                ((0,128,255), 'left'), ((0,128,255), 'right'), ((0,128,255), 'top'), ((0,128,255), 'bottom')]
elif args.env_name == "grid_illustrative_random":
    # Random training colours
    train_tasks = [((206,16,220), 'left'), ((206,16,220), 'right'), ((206,16,220), 'top'), ((206,16,220), 'bottom'),
                ((86,185,105), 'left'), ((86,185,105), 'right'), ((86,185,105), 'top'), ((86,185,105), 'bottom')]
    # Random testing colours
    test_tasks = [((237,48,217), 'left'), ((237,48,217), 'right'), ((237,48,217), 'top'), ((237,48,217), 'bottom'),
                ((52,128,100), 'left'), ((52,128,100), 'right'), ((52,128,100), 'top'), ((52,128,100), 'bottom'),
                ((35,21,88), 'left'), ((35,21,88), 'right'), ((35,21,88), 'top'), ((35,21,88), 'bottom'),
                ((213,109,113), 'left'), ((213,109,113), 'right'), ((213,109,113), 'top'), ((213,109,113), 'bottom')]
elif args.env_name == "control_illustrative":
    # C4 Rotations
    train_tasks = [(45,-45,0), (45,-45,90), (45,-45,180), (45,-45,270)]
    # Random testing rotations
    test_tasks = [(45,-45,-11), (45,-45,11), (45,-45,65), (45,-45,99), (45,-45,167), (45,-45,204), (45,-45,259), (45,-45,325)]
elif args.env_name == "control_illustrative_random":
    # Random Rotations
    train_tasks = [(45,-45,-6), (45,-45,94), (45,-45,171), (45,-45,289)]
    # Random testing rotations
    test_tasks = [(45,-45,-11), (45,-45,11), (45,-45,65), (45,-45,99), (45,-45,167), (45,-45,204), (45,-45,259), (45,-45,325)]
elif args.env_name == "control_illustrative_base":
    # Base tasks: same rotations as the random rotations above, but with randomised starting poses for the robot arm
    train_tasks = [
        (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
    ]
    test_tasks = []
elif args.env_name == "control_illustrative_da_expl":
    # Now we perform 'data augmentation' by enumerating all poses for all rotations for the base set of tasks above
    train_tasks = [
        (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
        (105,213,-6), (152,290,94), (358,43,171), (248,77,289),     # duplicate all the poses in all tasks
        (152,290,-6), (358,43,94), (248,77,171), (105,213,289),     # duplicate all the poses in all tasks
        (358,43,-6), (248,77,94), (105,213,171), (152,290,289),     # duplicate all the poses in all tasks
    ]
    test_tasks = []
elif args.env_name == "control_illustrative_random_expl":
    # Now we imitate random pure exploration at the start of each episode (ExploreGo)
    # by adding three random poses to each of the base tasks
    train_tasks = [
        (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
        (288,75,-6), (213,17,94), (56,215,171), (357,274,289),     # random poses in all tasks
        (40,86,-6), (305,136,94), (187,91,171), (101,240,289),     # random poses in all tasks
        (129,254,-6), (117,352,94), (60,167,171), (258,214,289),     # random poses in all tasks
    ]
    test_tasks = []
elif args.env_name == "control_illustrative_unreachable":
    # Now we add completely new (unreachable) tasks by randomly sampling from the full distribution
    train_tasks = [
        (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
        (288,75,20), (213,17,305), (56,215,269), (357,274,293),     # random poses and rotations in all tasks
        (40,86,185), (305,136,2), (187,91,350), (101,240,101),     # random poses and rotations in all tasks
        (129,254,22), (117,352,57), (60,167,118), (258,214,230),     # random poses and rotations in all tasks
    ]
    test_tasks = []
elif args.env_name == "control_illustrative_unreachable32":
    # Now we add completely new (unreachable) tasks by randomly sampling from the full distribution
    train_tasks = [
        (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
        (288,75,20), (213,17,305), (56,215,269), (357,274,293),     # random poses and rotations in all tasks
        (40,86,185), (305,136,2), (187,91,350), (101,240,101),     # random poses and rotations in all tasks
        (129,254,22), (117,352,57), (60,167,118), (258,214,230),     # random poses and rotations in all tasks
        (187,2,207), (337,91,143), (197,146 ,54), (138,293,200),     # random poses and rotations in all tasks
        (38,195,284), (78,259,1), (97,139,19), (109,53,261),     # random poses and rotations in all tasks
        (60,217,235), (39,252,162), (304,307,283), (264,216,240),     # random poses and rotations in all tasks
        (52,191,186), (164,179,149), (80,13,106), (124,95,265),     # random poses and rotations in all tasks
    ]
    test_tasks = []
else:
    set_id = args.env_name[-1]
    with open(f'datasets/task_sets_{set_id}.json', 'r') as file:
        tasks_dict = json.load(file)
    
    if "base" in args.env_name:
        train_tasks = tasks_dict['base_tasks']
    elif "da_expl" in args.env_name:
        train_tasks = tasks_dict['da_expl_tasks']
    elif "random_expl" in args.env_name:
        train_tasks = tasks_dict['random_expl_tasks']
    elif "unreachablex2" in args.env_name:
        train_tasks = tasks_dict['unreachablex2_tasks']
    elif "unreachablex4" in args.env_name:
        train_tasks = tasks_dict['unreachablex4_tasks']
    elif "unreachablex8" in args.env_name:
        train_tasks = tasks_dict['unreachablex8_tasks']
    test_tasks = []



if "grid" in args.env_name:
    env_kwargs = {"arm_length": 6}
    if "cont" in args.algo:
        env = gym.make('GridIllustrativeCMDPContinuous-v0', tasks=train_tasks, **env_kwargs)
        eval_env_type = "grid_continuous"
    else:
        env = gym.make('GridIllustrativeCMDPDiscrete-v0', tasks=train_tasks, **env_kwargs)
        eval_env_type = "grid_discrete"
elif "control" in args.env_name:
    env_kwargs = {}
    env = gym.make('ControlIllustrativeCMDP-v0', tasks=train_tasks)
    eval_env_type = "control"

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
            env=eval_env_type,
            eval_eps=args.eval_eps,
            env_kwargs=env_kwargs
        )
        train_mean_perf, train_mean_len = eval_agent(
            agent,
            device,
            train_tasks,
            env=eval_env_type,
            eval_eps=args.eval_eps,
            env_kwargs=env_kwargs
        )
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
            log_stats(stats_dict)

    # Save agent and number of epochs
    if args.resume and (epoch+1) % args.ckpt_freq == 0:
        curr_epochs = epoch + 1
        agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "model.pt"))
        agent.save(num_epochs=curr_epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, f"model_{epoch}.pt"))
                
test_mean_perf, test_mean_len = eval_agent(agent, device, test_tasks, env=eval_env_type, eval_eps=args.eval_eps, env_kwargs=env_kwargs)
train_mean_perf, train_mean_len = eval_agent(agent, device, train_tasks, env=eval_env_type, eval_eps=args.eval_eps, env_kwargs=env_kwargs)

wandb.log({"final_test_ret": test_mean_perf, "final_test_len": test_mean_len, "final_train_ret": train_mean_perf, "final_train_len": train_mean_len}, step=(epoch + 1))
filewriter.log_final_test_eval({
        'final_test_ret': test_mean_perf,
        'final_train_ret': train_mean_perf,
    })
if args.resume:
    agent.save(num_epochs=args.epochs, path=os.path.join(args.save_path, args.env_name, args.xpid, "final_model.pt"))
