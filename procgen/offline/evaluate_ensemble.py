import argparse
import logging
import os
import time
from copy import deepcopy

import procgen
import torch
import torch.nn as nn
from baselines.common.vec_env import VecExtractDictObs
from torch.utils.data import DataLoader

import wandb
from offline.agents import _create_agent
from offline.arguments import parser
from offline.test_offline_agent import eval_agent, eval_DT_agent
from utils.filewriter import FileWriter
from utils.utils import set_seed
from online.behavior_policies.distributions import FixedCategorical

class AggregateEnsemble():
    def __init__(self, agents):
        self.agents = agents

    def eval(self):
        [a.eval() for a in self.agents]

    def eval_step(self, observation, eps=0.0):
        """
        Given an observation, return an action.

        :param observation: the observation for the environment
        :return: the action for the environment in numpy
        """
        deterministic = eps == 0.0
        with torch.no_grad():
            actor_features = [a.model_base(observation) for a in self.agents]
            dists = [a.model_dist(actor_features[i]) for i, a in enumerate(self.agents)]
            
            # take the mean over the probs
            probs = torch.stack([d.probs for d in dists]).mean(dim=0)
            dist = FixedCategorical(probs=probs)
            
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample()

        return action.cpu().numpy()

t0 = time.time()

args = parser.parse_args()
print(args)

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

set_seed(args.seed)

if args.xpid is None:
    args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")

ensemble_size = args.ensemble_size
start_id = args.start_id
base_xpid = args.xpid
log_dir = os.path.expandvars(os.path.expanduser(os.path.join(args.save_path, args.env_name)))
final_result_dir = os.path.join(log_dir, base_xpid + f"_{start_id}", f"evaluate_aggregate_{ensemble_size}.csv")

if os.path.exists(final_result_dir):
    print("Final evaluate csv already exists in the log_dir")
    exit(0)

# create Procgen env
env = procgen.ProcgenEnv(num_envs=1, env_name=args.env_name)
env = VecExtractDictObs(env, "rgb")


# Initialize agent
agents = []
for i in range(ensemble_size):
    # check if final_model.pt already exists in the log_dir
    xpid = base_xpid + f"_{start_id+i}"
    if not os.path.exists(os.path.join(log_dir, xpid, "final_model.pt")):
        raise FileNotFoundError(f"Final model does not exist in the log_dir for xpid {xpid}")
    temp_args = deepcopy(args)
    temp_args.xpid = xpid
    agent = _create_agent(temp_args, env=env, extra_config={})
    agent.set_device(device)

    # load checkpoint and resume if resume flag is true
    curr_epochs = agent.load(os.path.join(args.save_path, args.env_name, xpid, "final_model.pt"))
    print(f"Checkpoint Loaded!")

    agents.append(agent)

agents = AggregateEnsemble(agents)
print("Models Created!")

test_mean_perf = eval_agent(agents, device, env_name=args.env_name, start_level=args.num_levels+50, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps, num_episodes=100)
train_mean_perf = eval_agent(
    agents, device, env_name=args.env_name, num_levels=args.num_levels, start_level=0, distribution_mode=args.distribution_mode, eval_eps=args.eval_eps, num_episodes=100
)
val_mean_perf = eval_agent(
    agents, device, env_name=args.env_name, num_levels=50, start_level=args.num_levels, distribution_mode=args.distribution_mode, num_episodes=100
)

# save dict to csv in logdir
with open(final_result_dir, "w") as f:
    f.write("final_test_ret,final_train_ret,final_val_ret\n")
    f.write(f"{test_mean_perf},{train_mean_perf},{val_mean_perf}\n")
    
print(f"Final Test Return: {test_mean_perf}")
print(f"Final Train Return: {train_mean_perf}")
print(f"Final Val Return: {val_mean_perf}")

print(f"Done in {time.time() - t0} seconds!")