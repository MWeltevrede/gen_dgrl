# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/idaac/blob/main/ppo_daac_idaac/model.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from online.behavior_policies.distributions import Categorical
from utils.utils import init

import numpy as np

init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

init_relu_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain("relu"))


def apply_init_(modules):
    """
    Initialize NN modules
    """
    for m in modules:
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Flatten(nn.Module):
    """
    Flatten a tensor
    """

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class Conv2d_tf(nn.Conv2d):
    """
    Conv2d with the padding behavior from TF
    """

    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get("padding", "SAME")

    def _compute_padding(self, input, dim):
        input_size = input.size(dim + 2)
        filter_size = self.weight.size(dim + 2)
        effective_filter_size = (filter_size - 1) * self.dilation[dim] + 1
        out_size = (input_size + self.stride[dim] - 1) // self.stride[dim]
        total_padding = max(0, (out_size - 1) * self.stride[dim] + effective_filter_size - input_size)
        additional_padding = int(total_padding % 2 != 0)

        return additional_padding, total_padding

    def forward(self, input):
        if self.padding == "VALID":
            return F.conv2d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=0,
                dilation=self.dilation,
                groups=self.groups,
            )
        rows_odd, padding_rows = self._compute_padding(input, dim=0)
        cols_odd, padding_cols = self._compute_padding(input, dim=1)
        if rows_odd or cols_odd:
            input = F.pad(input, [0, cols_odd, 0, rows_odd])

        return F.conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            padding=(padding_rows // 2, padding_cols // 2),
            dilation=self.dilation,
            groups=self.groups,
        )


class NNBase(nn.Module):
    """
    Actor-Critic network (base class)
    """

    def __init__(self, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size

    @property
    def output_size(self):
        return self._hidden_size


class BasicBlock(nn.Module):
    """
    Residual Network Block
    """

    def __init__(self, n_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = Conv2d_tf(n_channels, n_channels, kernel_size=3, stride=1, padding=(1, 1))
        self.stride = stride

        apply_init_(self.modules())

        self.train()

    def forward(self, x):
        identity = x

        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity
        return out


class ResNetBase(NNBase):
    """
    Residual Network
    """

    def __init__(self, num_inputs, hidden_size=256, channels=[16, 32, 32]):
        super(ResNetBase, self).__init__(hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=stride))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        return self.critic_linear(x), x


class PolicyResNetBase(NNBase):
    """
    Residual Network
    """

    def __init__(self, num_inputs, hidden_size=256, channels=[16, 32, 32], num_actions=15):
        super(PolicyResNetBase, self).__init__(hidden_size)
        self.num_actions = num_actions

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size + num_actions, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=stride))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs, actions=None):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        if actions is None:
            onehot_actions = torch.zeros(x.shape[0], self.num_actions).to(x.device)
        else:
            onehot_actions = F.one_hot(actions.squeeze(1), self.num_actions).float()
        gae_inputs = torch.cat((x, onehot_actions), dim=1)

        return self.critic_linear(gae_inputs), x


class ValueResNet(NNBase):
    """
    Residual Network
    """

    def __init__(self, num_inputs, hidden_size=256, channels=[16, 32, 32]):
        super(ValueResNet, self).__init__(hidden_size)

        self.layer1 = self._make_layer(num_inputs, channels[0])
        self.layer2 = self._make_layer(channels[0], channels[1])
        self.layer3 = self._make_layer(channels[1], channels[2])

        self.flatten = Flatten()
        self.relu = nn.ReLU()

        self.fc = init_relu_(nn.Linear(2048, hidden_size))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        apply_init_(self.modules())

        self.train()

    def _make_layer(self, in_channels, out_channels, stride=1):
        layers = []

        layers.append(Conv2d_tf(in_channels, out_channels, kernel_size=3, stride=stride))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        layers.append(BasicBlock(out_channels))
        layers.append(BasicBlock(out_channels))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        x = inputs

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.relu(self.flatten(x))
        x = self.relu(self.fc(x))

        return self.critic_linear(x)


class LinearOrderClassifier(nn.Module):
    def __init__(self, emb_size=256):
        super(LinearOrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_(nn.Linear(2 * emb_size, 2)),
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x


class NonlinearOrderClassifier(nn.Module):
    def __init__(self, emb_size=256, hidden_size=4):
        super(NonlinearOrderClassifier, self).__init__()
        self.main = nn.Sequential(
            Flatten(),
            init_relu_(nn.Linear(2 * emb_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, 2)),
            nn.Softmax(dim=1),
        )
        self.train()

    def forward(self, emb):
        x = self.main(emb)
        return x
    

class IllustrativeEncoder(NNBase):
    def __init__(self, observation_space, output_space, hidden_size=64, channels=[128, 64], use_actor_linear=True, normalize_obs=True):
        super().__init__(hidden_size)
        flattened_dim = np.prod(observation_space.shape)
        self.normalize_obs = normalize_obs

        self.linears = []
        self.linears.append(Flatten())
        self.linears.append(nn.Linear(flattened_dim, channels[0]))
        self.linears.append(nn.Tanh())
        for i in range(len(channels) - 1):
            self.linears.append(nn.Linear(channels[i], channels[i + 1]))
            self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(channels[-1], hidden_size))
        self.linears.append(nn.Tanh())
        self.linears.append(nn.Linear(hidden_size, output_space))
        self.linears = nn.Sequential(*self.linears)


    def forward(self, x):
        if self.normalize_obs:
            x = x / 255.
        return self.linears(x)


class PPOnet(nn.Module):
    """
    PPO netowrk
    """

    def __init__(self, obs_shape, action_space, initial_sd=.25, base_kwargs=None):
        super(PPOnet, self).__init__()

        if base_kwargs is None:
            base_kwargs = {}

        base = IllustrativeEncoder
        self.action_shape = action_space.shape[0]

        self.actor = base(obs_shape, 2*self.action_shape, **base_kwargs)
        self.initial_sd = torch.as_tensor(initial_sd, dtype=torch.float32).detach()
        self.critic = base(obs_shape, 1, **base_kwargs)

        with torch.no_grad():
            # force the weights of the last layer to be really small
            # this initializes the actions to be observation independent with a value of 0
            # which is a nice unbiased starting point for the policy network
            self.actor.linears[-1].weight *= 1/100

        self.low = torch.as_tensor(action_space.low).float()
        self.high = torch.as_tensor(action_space.high).float()

    def forward(self, inputs):
        raise NotImplementedError
    
    def set_device(self, device):
        self.actor.to(device)
        self.critic.to(device)
        self.initial_sd.to(device)
        self.low = self.low.to(device)
        self.high = self.high.to(device)

    def bound(self, x):
        # turn x from range [-infty, infty] to [self.low, self.high]
        x = torch.tanh(x)
        return ((x+1)/2.)*(self.high - self.low) + self.low
    
    def unbound(self, x):
        # turn x from range [self.low, self.high] to [-infty, infty]
        x = 2.* (x - self.low) / (self.high - self.low)
        x = x - 1
        x = torch.atanh(x)
        return x

    def act(self, inputs, deterministic=False):
        value, actor_features = self.critic(inputs), self.actor(inputs)
        sd_constant = self.initial_sd + torch.log(1 - torch.exp(-self.initial_sd))
        dist = Normal(actor_features[:, :self.action_shape], F.softplus(actor_features[:, self.action_shape:] + sd_constant))

        if deterministic:
            unbound_action = dist.mode()
        else:
            unbound_action = dist.sample()

        action = self.bound(unbound_action)

        action_log_probs = dist.log_prob(unbound_action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs):
        value = self.critic(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.critic(inputs), self.actor(inputs)
        sd_constant = self.initial_sd + torch.log(1 - torch.exp(-self.initial_sd))
        dist = Normal(actor_features[:, :self.action_shape], F.softplus(actor_features[:, self.action_shape:] + sd_constant))

        action_log_probs = dist.log_prob(self.unbound(action)).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy
