# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from control_illustrative_env import ControlIllustrativeVenv
from env_util import make_vec_env


def evaluate(args, model: nn.Module, device, num_episodes=10):
    model.eval()

    # Sample Levels From the Full Distribution
    eval_envs = ControlIllustrativeVenv(
        n_envs=1,
        device=device,
    )
    # eval_envs = make_vec_env("Pendulum-v0", 1, device='cuda')

    eval_episode_rewards = []
    obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _ = model.act(obs)

        obs, _reward, _done, infos = eval_envs.step(action)

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])

    eval_envs.close()
    return eval_episode_rewards
