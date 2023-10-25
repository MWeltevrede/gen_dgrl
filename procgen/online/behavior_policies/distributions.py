# Copyright (c) 2017 Ilya Kostrikov
# 
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://opensource.org/licenses/MIT
#
# This file is a modified version of:
# https://github.com/rraileanu/auto-drac/blob/master/ucb_rl2_meta/arguments.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import init


class FixedCategorical(torch.distributions.Categorical):
    """
    Categorical distribution object
    """

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)
    
    def _get_logits(self):
        return self.logits
    
    def _get_log_softmax(self):
        log_probs = F.log_softmax(self.logits, dim=1)
        return log_probs


class Categorical(nn.Module):
    """
    Categorical distribution (NN module)
    """

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)
