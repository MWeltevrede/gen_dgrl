# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from offline.agents.bc import BehavioralCloning, BehavioralCloningContinuous
from offline.agents.bc_n import BehavioralCloningEnsemble, BehavioralCloningEnsembleContinuous, ValueDistilEnsemble
from offline.agents.bcq import BCQ
from offline.agents.ddqn_cql import CQL
from offline.agents.iql import IQL, IQLEnsemble
from offline.agents.dt import DecisionTransformer

def _create_agent(args, env, extra_config):
	agent_name = args.algo
	if agent_name == "bc_cont":
		return BehavioralCloningContinuous(env.observation_space, env.action_space, args.lr, args.agent_model, hidden_size=args.hidden_size, channels=args.channels, normalize_obs=args.normalize_obs, activation=args.activation)
	if agent_name == "bc":
		return BehavioralCloning(env.observation_space, env.action_space.n, args.lr, args.agent_model, hidden_size=args.hidden_size, channels=args.channels, normalize_obs=args.normalize_obs, activation=args.activation)
	if agent_name == "bc_n_cont":
		return BehavioralCloningEnsembleContinuous(env.observation_space, env.action_space, args.lr, args.agent_model, ensemble_size=args.ensemble_size, hidden_size=args.hidden_size, channels=args.channels, normalize_obs=args.normalize_obs, activation=args.activation)
	if agent_name == "value_distil":
		return ValueDistilEnsemble(env.observation_space, env.action_space, args.lr, args.agent_model, ensemble_size=args.ensemble_size, hidden_size=args.hidden_size, channels=args.channels, normalize_obs=args.normalize_obs, activation=args.activation, policy_extraction=args.policy_extraction, policy_extraction_temp=args.policy_extraction_temp, policy_extraction_pessimism_coef=args.policy_extraction_pessimism_coef)
	if agent_name == "bc_n":
		return BehavioralCloningEnsemble(env.observation_space, env.action_space.n, args.lr, args.agent_model, ensemble_size=args.ensemble_size, hidden_size=args.hidden_size, channels=args.channels, normalize_obs=args.normalize_obs, activation=args.activation)
	if agent_name == "bcq":
		assert args.agent_model in ["bcq", "bcqresnetbase"]
		return BCQ(env.observation_space, 
				   env.action_space.n, 
				   args.lr, 
				   args.agent_model, 
				   args.hidden_size,
				   gamma=args.gamma,
				   target_update_freq=args.target_update_freq,
				   tau=args.tau,
				   eps_start=args.eps_start,
				   eps_end=args.eps_end,
				   eps_decay=args.eps_decay,
				   bcq_threshold=args.bcq_threshold,
				   perform_polyak_update=args.perform_polyak_update)
	elif agent_name == "cql":
		return CQL(env.observation_space, 
				   env.action_space.n, 
				   args.lr, 
				   args.agent_model, 
				   args.hidden_size,
				   channels=args.channels,
				   gamma=args.gamma,
				   target_update_freq=args.target_update_freq,
				   tau=args.tau,
				   eps_start=args.eps_start,
				   eps_end=args.eps_end,
				   eps_decay=args.eps_decay,
				   cql_alpha=args.cql_alpha,
				   perform_polyak_update=args.perform_polyak_update,
				   normalize_obs=args.normalize_obs, 
				   activation=args.activation,
				   ensemble_size=args.cql_ensemble_size)
	elif agent_name == "iql":
		return IQL(env.observation_space, 
				   env.action_space, 
				   args.lr, 
				   args.agent_model, 
				   args.hidden_size,
				   channels=args.channels,
				   gamma=args.gamma,
				   target_update_freq=args.target_update_freq,
				   tau=args.tau,
				   eps_start=args.eps_start,
				   eps_end=args.eps_end,
				   eps_decay=args.eps_decay,
				   iql_temperature=args.iql_temperature,
				   iql_expectile=args.iql_expectile,
				   perform_polyak_update=args.perform_polyak_update,
				   normalize_obs=args.normalize_obs, 
				   activation=args.activation)
	elif agent_name == "iql_ensemble":
		return IQLEnsemble(env.observation_space, 
				   env.action_space, 
				   args.lr, 
				   args.agent_model, 
				   args.hidden_size,
				   channels=args.channels,
				   gamma=args.gamma,
				   target_update_freq=args.target_update_freq,
				   tau=args.tau,
				   eps_start=args.eps_start,
				   eps_end=args.eps_end,
				   eps_decay=args.eps_decay,
				   iql_temperature=args.iql_temperature,
				   iql_expectile=args.iql_expectile,
				   perform_polyak_update=args.perform_polyak_update,
				   normalize_obs=args.normalize_obs, 
				   activation=args.activation,
				   value_ensemble_size=args.iql_value_ensemble_size,
				   actor_ensemble_size=args.iql_actor_ensemble_size,
				   use_value=args.iql_use_value,
				   critic_da=args.iql_critic_da,
				   critic_concistency_coef=args.iql_critic_concistency_coef,
				   actor_da=args.iql_actor_da,
				   actor_concistency_coef=args.iql_actor_concistency_coef,
				   avg_q=args.iql_avg_q,
				   extract_all_actions=args.iql_extract_all_actions, 
				   pessimism_coef=args.iql_pessimism_coef)
	elif agent_name in ["dt", "bct"]:
		return DecisionTransformer(env.observation_space,
								   env.action_space.n, 
									args.agent_model, 
									extra_config["train_data_vocab_size"],
									extra_config["train_data_block_size"],
									extra_config["max_timesteps"],
									args.dt_context_length,
									extra_config["dataset_size"],
									lr=args.lr)
	else:
		raise ValueError(f"Invalid agent name {agent_name}.")
