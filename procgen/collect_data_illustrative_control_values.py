from gym.envs.registration import register
import gym
import numpy as np
import io, os
import torch
import copy
import json
import random

from stable_baselines3 import SAC, DQN

def generate_symmetric_dataset(tasks, dataset_name, model, e_greedy=False):
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=3, simple_r_function=False)
	env.seed(87)
	obs = env.reset()

	dataset_dirname = f'datasets/{dataset_name}'
	os.makedirs(dataset_dirname, exist_ok=True)

	ep_obs = [obs]
	ep_rewards = []
	ep_dones = []
	ep_actions = []
	ep_values = []
	total_steps = 0
	total_episodes = len(tasks) if not e_greedy else 5*len(tasks)
	for i in range(total_episodes):
		if i % len(tasks) == 0:
			action_seq = []
		done = False
		step = 0
		while not done:
			act_obs = env.get_base_state()
			act_obs = torch.from_numpy(act_obs).float().to('cuda')
			if len(act_obs.shape) == 1:
				# add batch dimension
				act_obs = act_obs.unsqueeze(0)
			with torch.no_grad():
				#action = model.policy._predict(act_obs, deterministic=True)
				action = model.policy(act_obs)
				values = model.policy.q_net(act_obs)
			# using numpy, if action is of shape [1,1] then convert it to [1]
			action = action.cpu().numpy()
			if len(action.shape) == 2:
				action = action[0]

			# action = optimal_policy(step)
			#obs, reward, done, _ = env.step(np.array(action))
			if i % len(tasks) == 0 and e_greedy and np.random.rand() < 0.05:
				action = [np.random.randint(0, env.action_space.n)]
				print('Different action taken!')
			if i % len(tasks) == 0:
				action_seq.append(action)
			else:
				action = action_seq[step]
			obs, reward, done, _ = env.step(action[0])
			ep_obs.append(obs)
			ep_rewards.append([reward])
			ep_dones.append(done)
			ep_actions.append(action)
			ep_values.append(values.squeeze(0).cpu().numpy())
			step += 1

			if done:
				total_steps += step
				episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_values), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
				with io.BytesIO() as bs:
					np.savez_compressed(bs, **episode)
					bs.seek(0)
					with open(dataset_dirname + f'/episode_{i}.npy', "wb") as f:
						f.write(bs.read())
				obs = env.reset()
				ep_obs = [obs]
				ep_rewards = []
				ep_dones = []
				ep_actions = []
				ep_values = []

	return total_steps

def generate_pessimistic_symmetric_dataset(tasks, dataset_name, model, e_greedy=False):
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=3, simple_r_function=False)
	env.seed(88)
	obs = env.reset()

	dataset_dirname = f'datasets/{dataset_name}'
	os.makedirs(dataset_dirname, exist_ok=True)

	ep_obs = [obs]
	ep_rewards = []
	ep_dones = []
	ep_actions = []
	ep_values = []
	total_steps = 0
	total_episodes = len(tasks) if not e_greedy else 5*len(tasks)
	for i in range(total_episodes):
		if i % len(tasks) == 0:
			action_seq = []
		done = False
		step = 0
		while not done:
			act_obs = env.get_base_state()
			act_obs = torch.from_numpy(act_obs).float().to('cuda')
			if len(act_obs.shape) == 1:
				# add batch dimension
				act_obs = act_obs.unsqueeze(0)
			with torch.no_grad():
				#action = model.policy._predict(act_obs, deterministic=True)
				action = model.policy(act_obs)
				values = model.policy.q_net(act_obs)
			# using numpy, if action is of shape [1,1] then convert it to [1]
			action = action.cpu().numpy()
			if len(action.shape) == 2:
				action = action[0]

			values = values.squeeze(0).cpu().numpy()
			pessimism = np.ones(values.shape) * 0.005 * 7.5
			pessimism[action[0]] = 0
			pessimistic_values = values - pessimism

			# action = optimal_policy(step)
			#obs, reward, done, _ = env.step(np.array(action))
			if i % len(tasks) == 0 and e_greedy and np.random.rand() < 0.05:
				action = [np.random.randint(0, env.action_space.n)]
				print('Different action taken!')
			if i % len(tasks) == 0:
				action_seq.append(action)
			else:
				action = action_seq[step]
			obs, reward, done, _ = env.step(action[0])
			ep_obs.append(obs)
			ep_rewards.append([reward])
			ep_dones.append(done)
			ep_actions.append(action)
			ep_values.append(pessimistic_values)
			step += 1

			if done:
				total_steps += step
				episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_values), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
				with io.BytesIO() as bs:
					np.savez_compressed(bs, **episode)
					bs.seek(0)
					with open(dataset_dirname + f'/episode_{i}.npy', "wb") as f:
						f.write(bs.read())
				obs = env.reset()
				ep_obs = [obs]
				ep_rewards = []
				ep_dones = []
				ep_actions = []
				ep_values = []


	return total_steps

def generate_pessimistic_random_dataset(tasks, dataset_name, model, e_greedy=False):
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=3, simple_r_function=False)
	env.seed(88)
	obs = env.reset()

	dataset_dirname = f'datasets/{dataset_name}'
	os.makedirs(dataset_dirname, exist_ok=True)

	ep_obs = [obs]
	ep_rewards = []
	ep_dones = []
	ep_actions = []
	ep_values = []
	total_steps = 0
	traj_specific_pessimism = [.5, 5, 10, 15]
	#traj_specific_pessimism = [.5, 2.5, 7.5, 10]
	random.shuffle(traj_specific_pessimism)
	print(traj_specific_pessimism)
	total_episodes = len(tasks) if not e_greedy else 5*len(tasks)
	for i in range(total_episodes):
		if i % len(tasks) == 0:
			action_seq = []
		done = False
		step = 0
		while not done:
			act_obs = env.get_base_state()
			act_obs = torch.from_numpy(act_obs).float().to('cuda')
			if len(act_obs.shape) == 1:
				# add batch dimension
				act_obs = act_obs.unsqueeze(0)
			with torch.no_grad():
				#action = model.policy._predict(act_obs, deterministic=True)
				action = model.policy(act_obs)
				values = model.policy.q_net(act_obs)
			# using numpy, if action is of shape [1,1] then convert it to [1]
			action = action.cpu().numpy()
			if len(action.shape) == 2:
				action = action[0]

			values = values.squeeze(0).cpu().numpy()
			#pessimism = np.random.uniform(0, 2, size=values.shape) * 0.005		# Noise-like pessimism
			pessimism = np.ones(values.shape) * 0.005 * traj_specific_pessimism[int(i % len(tasks))]		# Trajectory-specific pessimism
			pessimism[action[0]] = 0
			pessimistic_values = values - pessimism

			# action = optimal_policy(step)
			#obs, reward, done, _ = env.step(np.array(action))
			if i % len(tasks) == 0 and e_greedy and np.random.rand() < 0.05:
				action = [np.random.randint(0, env.action_space.n)]
				print('Different action taken!')
			if i % len(tasks) == 0:
				action_seq.append(action)
			else:
				action = action_seq[step]
			obs, reward, done, _ = env.step(action[0])
			ep_obs.append(obs)
			ep_rewards.append([reward])
			ep_dones.append(done)
			ep_actions.append(action)
			ep_values.append(pessimistic_values)
			step += 1

			if done:
				total_steps += step
				episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_values), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
				with io.BytesIO() as bs:
					np.savez_compressed(bs, **episode)
					bs.seek(0)
					with open(dataset_dirname + f'/episode_{i}.npy', "wb") as f:
						f.write(bs.read())
				obs = env.reset()
				ep_obs = [obs]
				ep_rewards = []
				ep_dones = []
				ep_actions = []
				ep_values = []


	return total_steps

def generate_suboptimal_dataset(tasks, test_tasks, dataset_name, model, symmetric_data=False, symmetric_pessimism=False, be_pessimistic=False):
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=3, simple_r_function=False)
	env.seed(88)
	obs = env.reset()

	if be_pessimistic:
		dataset_dirname = f'datasets/suboptimal/{dataset_name}_symdat{symmetric_data}_sympes{symmetric_pessimism}'
	else:
		dataset_dirname = f'datasets/suboptimal/{dataset_name}_symdat{symmetric_data}'
	os.makedirs(dataset_dirname, exist_ok=True)

	ep_obs = [obs]
	ep_rewards = []
	ep_dones = []
	ep_actions = []
	ep_values = []
	total_steps = 0
	traj_specific_pessimism = [.5, 5, 10, 15]
	#traj_specific_pessimism = [.5, 2.5, 7.5, 10]
	avg_pessimism = 7.5
	random.shuffle(traj_specific_pessimism)
	print(traj_specific_pessimism)
	#total_episodes = len(tasks) if not e_greedy else 5*len(tasks)
	total_episodes = len(tasks)
	for i in range(total_episodes):
		if symmetric_data and i % len(tasks) == 0:
			action_seq = []
		suboptimal_i = np.random.randint(7, 28-7) 	# 28 is the optimal trajectory length
		done = False
		step = 0
		while not done:
			act_obs = env.get_base_state()
			act_obs = torch.from_numpy(act_obs).float().to('cuda')
			if len(act_obs.shape) == 1:
				# add batch dimension
				act_obs = act_obs.unsqueeze(0)
			with torch.no_grad():
				#action = model.policy._predict(act_obs, deterministic=True)
				action = model.policy(act_obs)
				values = model.policy.q_net(act_obs)
			# using numpy, if action is of shape [1,1] then convert it to [1]
			action = action.cpu().numpy()
			if len(action.shape) == 2:
				action = action[0]

			if step >= suboptimal_i and step < suboptimal_i + 4:		# We do 4 suboptimal steps in a row
				sub_action = np.random.randint(0, env.action_space.n)
				while sub_action == action[0]:
					sub_action = np.random.randint(0, env.action_space.n)
				action = [sub_action]

			if symmetric_data:
				if i % len(tasks) == 0:
					action_seq.append(action)
				else:
					action = action_seq[step]

			values = values.squeeze(0).cpu().numpy()
			if be_pessimistic:
				if symmetric_pessimism:
					pessimism = np.ones(values.shape) * 0.005 * avg_pessimism
				else:
					pessimism = np.ones(values.shape) * 0.005 * traj_specific_pessimism[int(i % len(tasks))]	
			else:
				pessimism = np.zeros(values.shape)
			pessimism[action[0]] = 0
			pessimistic_values = values - pessimism

			obs, reward, done, _ = env.step(action[0])
			ep_obs.append(obs)
			ep_rewards.append([reward])
			ep_dones.append(done)
			ep_actions.append(action)
			ep_values.append(pessimistic_values)
			step += 1

			if done:
				total_steps += step
				episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_values), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
				with io.BytesIO() as bs:
					np.savez_compressed(bs, **episode)
					bs.seek(0)
					with open(dataset_dirname + f'/episode_{i}.npy', "wb") as f:
						f.write(bs.read())
				obs = env.reset()
				ep_obs = [obs]
				ep_rewards = []
				ep_dones = []
				ep_actions = []
				ep_values = []

	if be_pessimistic:
		dataset_sizes_filename = f'datasets/suboptimal/dataset_sizes_symdat{symmetric_data}_sympes{symmetric_pessimism}.txt'
		task_sets_filename = f"datasets/suboptimal/task_sets_symdat{symmetric_data}_sympes{symmetric_pessimism}.json"
	else:
		dataset_sizes_filename = f'datasets/suboptimal/dataset_sizes_symdat{symmetric_data}.txt'
		task_sets_filename = f"datasets/suboptimal/task_sets_symdat{symmetric_data}.json"


	with open(dataset_sizes_filename, 'w') as file:
		print(f"base: {total_steps}", file=file)
	with open(task_sets_filename, 'w') as file:
		json.dump({'base':tasks, 'test':test_tasks}, file)

	return total_steps


def generate_equivariant_dataset(tasks, dataset_name, model, e_greedy=False, det_action=False):
	# suboptimal actions (that appear to be suboptimal on all training tasks):
	# [0,1,3,4,6]
	# equi rotates dims 0 and 4
	# equi2 rotates dims 1 and 3
	# equi3 rotates 0 and 1
	# equi4 rotates 3 and 6
	# equi5 rotates 3 and 4
	# equi_pes rotates 3 and 6 and adds pessimism 0.005 * 0.5
	# equi_pes2 rotates 3 and 6 and adds pessimism 0.005 * 1
	# equi_pes3 rotates 3 and 6 and adds pessimism 0.005 * 5
	rot_dim1 = 0
	rot_dim2 = 4
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=3, simple_r_function=False)
	env.seed(88)
	obs = env.reset()

	dataset_dirname = f'datasets/{dataset_name}'
	os.makedirs(dataset_dirname, exist_ok=True)

	ep_obs = [obs]
	ep_rewards = []
	ep_dones = []
	ep_actions = []
	ep_values = []
	total_steps = 0
	total_episodes = len(tasks) if not e_greedy else 5*len(tasks)
	for i in range(total_episodes):
		if i % len(tasks) == 0:
			action_seq = []
		done = False
		step = 0
		while not done:
			act_obs = env.get_base_state()
			act_obs = torch.from_numpy(act_obs).float().to('cuda')
			if len(act_obs.shape) == 1:
				# add batch dimension
				act_obs = act_obs.unsqueeze(0)
			with torch.no_grad():
				#action = model.policy._predict(act_obs, deterministic=True)
				action = model.policy(act_obs)
				values = model.policy.q_net(act_obs)
			# using numpy, if action is of shape [1,1] then convert it to [1]
			action = action.cpu().numpy()
			if len(action.shape) == 2:
				action = action[0]

			values = values.squeeze(0).cpu().numpy()
			pessimism = np.zeros(values.shape)
			#pessimism = np.ones(values.shape) * 0.005 * 5
			#pessimism[action[0]] = 0
			pessimistic_values = values - pessimism
			if int(i % len(tasks)) == 1:
				# manually apply 90 degree rotation matrix on the 0 and 4 axis
				#pessimistic_values = values
				temp = pessimistic_values[rot_dim1]
				pessimistic_values[rot_dim1] = -pessimistic_values[rot_dim2]
				pessimistic_values[rot_dim2] = temp
			elif int(i % len(tasks)) == 2:
				# manually apply 180 degree rotation matrix on the 0 and 4 axis
				#pessimistic_values = values
				pessimistic_values[rot_dim1] = -pessimistic_values[rot_dim1]
				pessimistic_values[rot_dim2] = -pessimistic_values[rot_dim2]
			elif int(i % len(tasks)) == 3:
				# manually apply 270 degree rotation matrix on the 0 and 4 axis
				#pessimistic_values = values
				temp = pessimistic_values[rot_dim1]
				pessimistic_values[rot_dim1] = pessimistic_values[rot_dim2]
				pessimistic_values[rot_dim2] = -temp

			# action = optimal_policy(step)
			#obs, reward, done, _ = env.step(np.array(action))
			if i % len(tasks) == 0 and e_greedy and np.random.rand() < 0.05:
				action = [np.random.randint(0, env.action_space.n)]
				print('Different action taken!')
			if i % len(tasks) == 0:
				action_seq.append(action)
			else:
				action = action_seq[step]
			obs, reward, done, _ = env.step(action[0])
			ep_obs.append(obs)
			ep_rewards.append([reward])
			ep_dones.append(done)
			ep_actions.append(action)
			if det_action:
				ep_values.append([np.argmax(pessimistic_values)])
			else:
				ep_values.append(pessimistic_values)
			step += 1

			if done:
				total_steps += step
				episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_values), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
				with io.BytesIO() as bs:
					np.savez_compressed(bs, **episode)
					bs.seek(0)
					with open(dataset_dirname + f'/episode_{i}.npy', "wb") as f:
						f.write(bs.read())
				obs = env.reset()
				ep_obs = [obs]
				ep_rewards = []
				ep_dones = []
				ep_actions = []
				ep_values = []


	return total_steps

### Randomly generated
for set_id in range(1):
	np.random.seed(set_id)

	#num_tasks = 10
	#tasks = []
	#training_poses = set()
	#for _ in range(num_tasks):
	#	random_pose = (np.random.randint(0, 360), np.random.randint(0,360))
	#	while random_pose in training_poses:
	#		random_pose = (np.random.randint(0, 360), np.random.randint(0,360))
		
	#	training_poses.add(random_pose)
	#	# We create a full data augmentation dataset for C_4 (testing is also done for C_4)
	#	for shoulder_pos in [0, 90, 180, 270]:
	#		tasks.append([random_pose[0], random_pose[1], shoulder_pos])
	#	#tasks.append([random_pose[0], random_pose[1], 0])
	tasks = []
	for shoulder_pos in [0, 90, 180, 270]:
		tasks.append([45, -45, shoulder_pos])
	

	#num_test_tasks = 25
	#test_poses = set()
	#test_tasks = []
	#for _ in range(num_test_tasks):
	#	random_pose = (np.random.randint(0, 360), np.random.randint(0,360))
	#	while random_pose in training_poses or random_pose in test_poses:
	#		random_pose = (np.random.randint(0, 360), np.random.randint(0,360))
		
	#	test_poses.add(random_pose)
	#	for shoulder_pos in [0, 90, 180, 270]:
	#		test_tasks.append([random_pose[0], random_pose[1], shoulder_pos])
	num_test_tasks = 100
	test_tasks = []
	test_angles = set()
	for _ in range(num_test_tasks):
		random_angle = np.random.randint(0,360)
		while random_angle in test_angles or random_angle in [0,90,180,270]:
			random_angle = np.random.randint(0,360)
		test_angles.add(random_angle)

		test_tasks.append([45, -45, random_angle])
	
	register(
		id="ControlIllustrativeCMDP-v0",
		entry_point="control_illustrative_env:ControlIllustrativeCMDP",
	)
		
	model = DQN.load("dqn_control_illustrative")

	#base_size = generate_symmetric_dataset(tasks, f'control_illustrative_base', model, e_greedy=True)
	#base_size = generate_pessimistic_symmetric_dataset(tasks, f'control_illustrative_sym_pes3', model, e_greedy=False)
	#base_size = generate_pessimistic_random_dataset(tasks, f'control_illustrative_traj_spec_pes2', model, e_greedy=True)
	base_size = generate_equivariant_dataset(tasks, f'control_illustrative_equi_det', model, e_greedy=True, det_action=True)

	#base_size = generate_suboptimal_dataset(tasks, test_tasks, f'control_illustrative', model, symmetric_data=True, symmetric_pessimism=True, be_pessimistic=True)


	with open(f'datasets/dataset_sizes_equi_det.txt', 'w') as file:
	#with open(f'datasets/dataset_sizes_traj_spec_pes2.txt', 'w') as file:
	#with open(f'datasets/dataset_sizes_base.txt', 'w') as file:
		print(f"base: {base_size}", file=file)

	# save tasks
	with open(f"datasets/task_sets_equi_det.json", 'w') as file:
	#with open(f"datasets/task_sets_traj_spec_pes2.json", 'w') as file:
	#with open(f"datasets/task_sets_base.json", 'w') as file:
		json.dump({'base':tasks, 'test':test_tasks}, file)