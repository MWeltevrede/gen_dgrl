from gym.envs.registration import register
import gym
import numpy as np
import io, os
import torch
import copy
import json

from stable_baselines3 import SAC, DQN

def generate_symmetric_dataset(tasks, dataset_name, model):
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
	for i in range(len(tasks)):
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
		i += 1

	return total_steps




### Randomly generated
for set_id in range(1):
	# set_id = 1
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

	#base_size = generate_symmetric_dataset(tasks, f'control_illustrative_base_{set_id}', model)
	base_size = generate_symmetric_dataset(tasks, f'control_illustrative_base', model)
	#with open(f'datasets/dataset_sizes_random_poses_{set_id}.txt', 'w') as file:
	with open(f'datasets/dataset_sizes_base.txt', 'w') as file:
		print(f"base: {base_size}", file=file)
		#print(f"base + da: {da_size}", file=file)
		#print(f"base + random: {random_size}", file=file)

	# save tasks
	#with open(f"datasets/task_sets_{set_id}.json", 'w') as file:
	with open(f"datasets/task_sets_base.json", 'w') as file:
		#json.dump({'base':base_tasks.tolist(), 'base + da':base_da_tasks.tolist(), 'base + random':base_random_tasks.tolist()}, file)
		json.dump({'base':tasks, 'test':test_tasks}, file)
