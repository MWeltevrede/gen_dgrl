from gym.envs.registration import register
import gym
import numpy as np
import io, os
import torch
import copy
import json
import random
import imageio
from PIL import Image, ImageDraw, ImageFont

from stable_baselines3 import SAC, DQN

def render_image(env, action, reward):
	im_array = env.render()
	image = Image.fromarray((im_array*255.).astype('uint8'))
	image = image.resize((300,300))

	draw = ImageDraw.Draw(image)
	font = ImageFont.load_default()

	# Draw the text on the image
	if isinstance(action, int):
		draw.text((0,0), str(action), fill=(0, 0, 0), font=font)  # White text
	else:
		draw.text((0,0), str(['{:.1f}'.format(x) for x in action]), fill=(0, 0, 0), font=font)  # White text
	
	draw.text((1,10), '{:.5f}'.format(reward), fill=(0, 0, 0), font=font)  # White text

	# Convert the image back to an RGB array
	rgb_array_with_text = np.array(image) / 255.
	#rgb_array_with_text = image

	return rgb_array_with_text


def generate_suboptimal_symmetric_dataset(tasks, test_tasks, dataset_name, model, set_id, epsilon=0.4, render=True, symmetric=True):
	#env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=None, simple_r_function=False, epsilon=0.02, terminal=False)
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=3, simple_r_function=True, epsilon=0.02, terminal=True)
	env.seed(88)
	obs = env.reset()
	if render:
		images = []
		img = render_image(env, [0,0], 0)
		images.append(img)

	dataset_dirname = f'datasets/test/{dataset_name}'
	os.makedirs(dataset_dirname, exist_ok=True)

	ep_obs = [obs]
	ep_rewards = []
	ep_dones = []
	ep_actions = []
	ep_base_state = []
	total_steps = 0
	total_episodes = 4*len(tasks)
	images = []
	action_seq_0 = []
	action_seq_1 = []
	action_seq_2 = []
	action_seq_3 = []
	symmetric_biased_actions = np.random.choice([0,2,6,8], size=3, replace=False)
	for i in range(total_episodes):
		if i % len(tasks) == 0:
			action_seq = []
		done = False
		step = 0
		while not done:
			base_state = env.get_base_state()
			#act_obs = torch.from_numpy(obs).float().to('cuda')
			act_obs = torch.from_numpy(base_state).float().to('cuda')
			if len(act_obs.shape) == 1:
				# add batch dimension
				act_obs = act_obs.unsqueeze(0)
			with torch.no_grad():
				#action = model.policy._predict(act_obs, deterministic=True)
				#action = action.cpu().numpy()
				#if len(action.shape) == 2:
				#	action = action[0]
				action = model.policy(act_obs).item()
			# using numpy, if action is of shape [1,1] then convert it to [1]
			

			#if np.random.rand() < epsilon:	
			#	action = env.action_space.sample()
			if symmetric:
				#if i / len(tasks) >= 1 and step < 100:
				#	action = env.action_space.sample()
				#if i // len(tasks) in [1,2] and step < 100:
				#	action = env.action_space.sample()
				#elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#	action = env.action_space.sample()

				## symmetric_bias_discrete (epsilon=0.4) and symmetric_bias_discrete_2 (epsilon=0.6)
				#if i // len(tasks) in [1]:
				#	if step == 0:
				#		if np.random.rand() < 0.5:
				#			biased_action = 0
				#		else:
				#			biased_action = 8
				#	if np.random.rand() < epsilon:	
				#		action = biased_action
				#elif i // len(tasks) == 2 and step > -1:
				#	if step == 0:
				#		if np.random.rand() < 0.5:
				#			biased_action = 0
				#		else:
				#			biased_action = 8
				#	if np.random.rand() < epsilon:	
				#		action = biased_action
				#elif i // len(tasks) == 3 and step > -1:
				#	if step == 0:
				#		if np.random.rand() < 0.5:
				#			biased_action = 0
				#		else:
				#			biased_action = 8
				#	if np.random.rand() < epsilon:	
				#		action = biased_action

				# symmetric_bias_discrete_2_test (epsilon=0.4)
				if i // len(tasks) in [1, 2, 3]:
					if np.random.rand() < epsilon:	
						action = symmetric_biased_actions[ i // len(tasks) - 1].item()


			else:
				## non_symmetric_4traj_fullr
				#if (i % len(tasks)) in [1]:
				#	if i // len(tasks) in [3] and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [2]:
				#	if i // len(tasks) in [2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [3]:
				#	if i // len(tasks) in [1,2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()

				## non_symmetric_discrete
				#if (i % len(tasks)) in [1]:
				#	if i // len(tasks) in [3] and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [2]:
				#	if i // len(tasks) in [2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [3]:
				#	if i // len(tasks) in [1,2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()
						
				## non_symmetric_45_discrete
				#if (i % len(tasks)) in [1,6]:
				#	if i // len(tasks) in [3] and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [2,3]:
				#	if i // len(tasks) in [2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [5,7]:
				#	if i // len(tasks) in [1,2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()

				## non_symmetric_45_discrete_2
				#if (i % len(tasks)) in [1,6]:
				#	if i // len(tasks) in [3] and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [2,3,0]:
				#	if i // len(tasks) in [2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [4,5,7]:
				#	if i // len(tasks) in [0,1,2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()

				## non_symmetric_45_discrete_3
				#if (i % len(tasks)) in [2,7]:
				#	if i // len(tasks) in [3] and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [6,3,0,5]:
				#	if i // len(tasks) in [1,2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [1]:
				#	if i // len(tasks) in [0,1,2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()


				## non_symmetric_discrete_2
				#if (i % len(tasks)) in [2,3]:
				#	if i // len(tasks) in [1,2] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 3 and (step > 15 and step < 120):
				#		action = env.action_space.sample()

				## non_symmetric_discrete_3
				#if (i % len(tasks)) in [2,3]:
				#	if i // len(tasks) in [0,1] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) in [2,3] and (step > 15 and step < 120):
				#		action = env.action_space.sample()

				## non_symmetric_discrete_4
				#if (i % len(tasks)) in [2,3]:
				#	if i // len(tasks) in [1,2,3]:
				#		if np.random.rand() < 0.5:	
				#			action = env.action_space.sample()	


				# non_symmetric_3traj
				#if (i % len(tasks)) in [2,3]:
				#	if i // len(tasks) == 1 and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 2 and (step > 20 and step < 120):
				#		action = env.action_space.sample()


				# non_symmetric_3traj_2
				#if (i % len(tasks)) in [1,2,3]:
				#	if i // len(tasks) == 1 and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 2 and (step > 20 and step < 120):
				#		action = env.action_space.sample()

				## non_symmetric_3traj_6
				#if (i % len(tasks)) in [1]:
				#	if i // len(tasks) == 2 and (step > 20 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [2]:
				#	if i // len(tasks) in [1] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 2 and (step > 20 and step < 120):
				#		action = env.action_space.sample()
				#if (i % len(tasks)) in [3]:
				#	if i // len(tasks) in [0, 1] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 2 and (step > 20 and step < 120):
				#		action = env.action_space.sample()

				## non_symmetric_3traj_4
				#if (i % len(tasks)) in [2,3]:
				#	if i // len(tasks) in [0, 1] and step < 100:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) == 2 and (step > 20 and step < 120):
				#		action = env.action_space.sample()


				## non_symmetric_3traj_5
				#if (i % len(tasks)) in [2,3]:
				#	if i // len(tasks) in [0]:
				#		action = env.action_space.sample()	
				#	elif i // len(tasks) in [1,2] and step > 15:
				#		action = env.action_space.sample()

				## non_symmetric_bias_discrete
				#if (i % len(tasks)) in [1]:
				#	biased_action = 0
				#	if i // len(tasks) in [1]:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	elif i // len(tasks) == 2:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	elif i // len(tasks) == 3:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#if (i % len(tasks)) in [2]:
				#	biased_action = 8
				#	if i // len(tasks) in [1]:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	elif i // len(tasks) == 2:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	elif i // len(tasks) == 3:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#if (i % len(tasks)) in [3]:
				#	biased_action = 0
				#	if i // len(tasks) in [1]:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	elif i // len(tasks) == 2:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	elif i // len(tasks) == 3:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action

				# non_symmetric_bias_discrete_2 (epsilon=0.4) and non_symmetric_bias_discrete_3 (epsilon=0.3) and non_symmetric_bias_discrete_4 (epsilon=0.6)
				if (i % len(tasks)) in [0]:
					biased_action = 0
					if i // len(tasks) in [1,2,3]:
						if np.random.rand() < epsilon:	
							action = biased_action
				if (i % len(tasks)) in [1]:
					biased_action = 2
					if i // len(tasks) in [1,2,3]:
						if np.random.rand() < epsilon:	
							action = biased_action
				if (i % len(tasks)) in [2]:
					biased_action = 8
					if i // len(tasks) in [1,2,3]:
						if np.random.rand() < epsilon:	
							action = biased_action
				if (i % len(tasks)) in [3]:
					biased_action = 6
					if i // len(tasks) in [1,2,3]:
						if np.random.rand() < epsilon:	
							action = biased_action

				## non_symmetric_bias_discrete_5 (epsilon=0.4)
				#if (i % len(tasks)) in [0]:
				#	biased_action = 0
				#	if i // len(tasks) in [1,2,3]:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	if i // len(tasks) == 1:
				#		action_seq_0.append(action)
				#if (i % len(tasks)) in [1]:
				#	biased_action = 2
				#	if i // len(tasks) in [1,2,3]:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	if i // len(tasks) == 1:
				#		action_seq_1.append(action)
				#if (i % len(tasks)) in [2]:
				#	biased_action = 8
				#	if i // len(tasks) in [1,2,3]:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	if i // len(tasks) == 1:
				#		action_seq_2.append(action)
				#if (i % len(tasks)) in [3]:
				#	biased_action = 6
				#	if i // len(tasks) in [1,2,3]:
				#		if np.random.rand() < epsilon:	
				#			action = biased_action
				#	if i // len(tasks) == 1:
				#		action_seq_3.append(action)



			if symmetric and i % len(tasks) == 0:
				action_seq.append(action)
			else:
				if symmetric:
					action = action_seq[step]
				else:
					if (i % len(tasks)) in [0] and len(action_seq_0) > 0 and i // len(tasks) > 1:
						action = action_seq_0[step]
					if (i % len(tasks)) in [1] and len(action_seq_1) > 0 and i // len(tasks) > 1:
						action = action_seq_1[step]
					if (i % len(tasks)) in [2] and len(action_seq_2) > 0 and i // len(tasks) > 1:
						action = action_seq_2[step]
					if (i % len(tasks)) in [3] and len(action_seq_3) > 0 and i // len(tasks) > 1:
						action = action_seq_3[step]
			obs, reward, done, _ = env.step(action)
			ep_obs.append(obs)
			ep_rewards.append([reward])
			ep_dones.append(done)
			ep_actions.append(action)
			ep_base_state.append(base_state)
			step += 1
			if render and i % 4 == 0:
				img = render_image(env, action, reward)
				images.append(img)

			if done:
				total_steps += step
				if len(ep_actions) == 200:
					print(len(ep_actions))
				episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_actions), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones), 'base_state': np.array(ep_base_state)}
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
				ep_base_state = []
				if render and i % 4 == 0:
					img = render_image(env, [0,0], 0)
					images.append(img)

	if symmetric:
		name_add = "symmetric_bias_discrete_2_test"
	else:
		name_add = "non_symmetric_bias_discrete_2_test"
	#dataset_sizes_filename = f'datasets/suboptimal/data_symmetry/dataset_sizes_eps{epsilon}_{name_add}_{set_id}.txt'
	#task_sets_filename = f"datasets/suboptimal/data_symmetry/task_sets_eps{epsilon}_{name_add}_{set_id}.json"
	dataset_sizes_filename = f'datasets/test/dataset_sizes_{name_add}_{set_id}.txt'
	task_sets_filename = f"datasets/test/task_sets_{name_add}_{set_id}.json"


	with open(dataset_sizes_filename, 'w') as file:
		print(f"base: {total_steps}", file=file)
	with open(task_sets_filename, 'w') as file:
		json.dump({'base':tasks, 'test':test_tasks}, file)

	if render:
		imageio.mimsave(f"datasets/test/render.gif", [np.array(img*255, dtype=np.uint8) for i, img in enumerate(images) if i%1 == 0], duration=100)


	return total_steps


def generate_suboptimal_dataset(tasks, test_tasks, dataset_name, model, epsilon=0.25, render=True):
	env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks, n_actions=None, simple_r_function=True, epsilon=0.01, terminal=True)
	env.seed(88)
	obs = env.reset()
	if render:
		images = []
		img = render_image(env, [0,0], 0)
		images.append(img)

	dataset_dirname = f'datasets/suboptimal/data_symmetry/{dataset_name}'
	os.makedirs(dataset_dirname, exist_ok=True)

	ep_obs = [obs]
	ep_rewards = []
	ep_dones = []
	ep_actions = []
	ep_base_state = []
	total_steps = 0
	total_episodes = 10*len(tasks)
	images = []
	for i in range(total_episodes):
		done = False
		step = 0
		while not done:
			base_state = env.get_base_state()
			act_obs = torch.from_numpy(obs).float().to('cuda')
			if len(act_obs.shape) == 1:
				# add batch dimension
				act_obs = act_obs.unsqueeze(0)
			with torch.no_grad():
				action = model.policy._predict(act_obs, deterministic=True)
				#action = model.policy(act_obs).item()
			# using numpy, if action is of shape [1,1] then convert it to [1]
			action = action.cpu().numpy()
			if len(action.shape) == 2:
				action = action[0]

			if np.random.rand() < epsilon:	
				action = env.action_space.sample()

			obs, reward, done, _ = env.step(action)
			ep_obs.append(obs)
			ep_rewards.append([reward])
			ep_dones.append(done)
			ep_actions.append(action)
			ep_base_state.append(base_state)
			step += 1
			if render and i % 4 == 0:
				img = render_image(env, action, reward)
				images.append(img)

			if done:
				total_steps += step
				if len(ep_actions) == 200:
					print(len(ep_actions))
				episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_actions), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones), 'base_state': np.array(ep_base_state)}
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
				ep_base_state = []
				if render and i % 4 == 0:
					img = render_image(env, [0,0], 0)
					images.append(img)

	dataset_sizes_filename = f'datasets/suboptimal/data_symmetry/dataset_sizes_eps{epsilon}_single.txt'
	task_sets_filename = f"datasets/suboptimal/data_symmetry/task_sets_eps{epsilon}_single.json"


	with open(dataset_sizes_filename, 'w') as file:
		print(f"base: {total_steps}", file=file)
	with open(task_sets_filename, 'w') as file:
		json.dump({'base':tasks, 'test':test_tasks}, file)

	if render:
		imageio.mimsave(f"datasets/suboptimal/data_symmetry/render.gif", [np.array(img*255, dtype=np.uint8) for i, img in enumerate(images) if i%1 == 0], duration=100)


	return total_steps


### Randomly generated
for set_id in range(50, 100):
	np.random.seed(set_id)
	#np.random.seed(88 + set_id)

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
	#angles = [0, 45, 90, 135, 180, 225, 270, 315]
	angles = [0, 90, 180, 270]
	for shoulder_pos in angles:
	#for shoulder_pos in [90]:
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
		while random_angle in test_angles or random_angle in angles:
			random_angle = np.random.randint(0,360)
		test_angles.add(random_angle)

		test_tasks.append([45, -45, random_angle])
	
	register(
		id="ControlIllustrativeCMDP-v0",
		entry_point="control_illustrative_env:ControlIllustrativeCMDP",
	)
		
	model = DQN.load("dqn_control_illustrative")
	#model = SAC.load("sac_control_illustrative_non_terminal")
	#model = SAC.load("sac_control_illustrative_full")
	#model = DQN.load("dqn_control_illustrative_2")

	#epsilon = 0.5
	#base_size = generate_suboptimal_symmetric_dataset(tasks, test_tasks, f'control_illustrative_eps{epsilon}_symmetric_{set_id}', model, set_id, epsilon=epsilon, symmetric=True)
	#base_size = generate_suboptimal_symmetric_dataset(tasks, test_tasks, f'control_illustrative_eps{epsilon}_non_symmetric_{set_id}', model, set_id, epsilon=epsilon, symmetric=False)
	base_size = generate_suboptimal_symmetric_dataset(tasks, test_tasks, f'control_illustrative_symmetric_bias_discrete_2_test_{set_id}', model, set_id, symmetric=True)
	base_size = generate_suboptimal_symmetric_dataset(tasks, test_tasks, f'control_illustrative_non_symmetric_bias_discrete_2_test_{set_id}', model, set_id, symmetric=False)


	#with open(f'datasets/dataset_sizes_equi_det.txt', 'w') as file:
	##with open(f'datasets/dataset_sizes_traj_spec_pes2.txt', 'w') as file:
	##with open(f'datasets/dataset_sizes_base.txt', 'w') as file:
	#	print(f"base: {base_size}", file=file)

	## save tasks
	#with open(f"datasets/task_sets_equi_det.json", 'w') as file:
	##with open(f"datasets/task_sets_traj_spec_pes2.json", 'w') as file:
	##with open(f"datasets/task_sets_base.json", 'w') as file:
	#	json.dump({'base':tasks, 'test':test_tasks}, file)