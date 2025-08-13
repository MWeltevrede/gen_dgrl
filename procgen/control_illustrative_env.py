import numpy as np
import torch

import gym
from gym import spaces

TIMEOUT_STEPS = 200

def segment_distance(p, p1, p2):
	'''
		Find the distance between an N-dimensional point and a line
	segment. 
	Code from https://github.com/madphysicist/haggis/blob/992fd229799f90fef44d00cbb62820fc8a84bcf4/src/haggis/math.py#L571
	'''
	seg = p2 - p1
	norm2_seg = (seg * seg).sum(keepdims=True)
	t = ((p - p1) * seg).sum(keepdims=True) / norm2_seg
	p0 = p1 + t * seg


	dist = np.empty_like(p0)
	mask1 = t < 0
	mask2 = t > 1
	np.subtract(p0, p, where=~(mask1 | mask2), out=dist)
	np.subtract(p1, p, where=mask1, out=dist)
	np.subtract(p2, p, where=mask2, out=dist)

	dist = np.square(dist, out=dist).sum(keepdims=True)
	np.sqrt(dist, out=dist)

	return dist
	

class ControlIllustrativeCMDP(gym.Env):
	metadata = {"render_modes": ["rgb_array"], "render_fps": 4}
	'''
	A continuous control illustrative environment where the goal is to move a 2D robot arm such
	that the hand is at (0,0). 

	The task space is defined as follows:
	(shoulder angle, elbow angle, shoulder location along unit circle)
	- shoulder angle: the angle between the shoulder and elbow,
						which is 0 degrees when a line through the shoulder and elbow passes through (0,0).
						Defined as positive counter-clockwise.
	- elbow angle: the angle between the elbow and the hand,
						which is 0 degrees when a line through the elbow and hand passes through (0,0).
						Defined as positive counter-clockwise.
	- shoulder location along unit circle: the angle that defines the position of the shoulder along the unit circle. 
											Zero degrees is equal to position (0,1). Defined as positive counter-clockwise.
	'''
	def __init__(self, tasks=None, g=10.0, n_actions=None, simple_r_function=False, **kwargs):
		self.tasks = tasks
		self._current_task_id = 0
		if tasks is None:
			self.num_tasks = 0
		elif isinstance(tasks, int):
			self.num_tasks = 0
		else:
			self.num_tasks = len(tasks)
		self._target_location = np.array([0.0,0.0])
		self._epsilon = 0.01
		self.step_counter = 0
		self.simple_r_function = simple_r_function

		# self._target_region = []
		# for dx in np.linspace(-0.1, 0.1, 50):
		#     for dy in np.linspace(-0.1, 0.1, 50):
		#         if np.linalg.norm(np.array([dx, dy])) < self._epsilon:
		#             self._target_region.append(np.array([dx, dy]))

		self.max_speed = 4
		self.max_torque = 2.0
		self.dt = 0.05
		self.g = g
		self.m = 1.0
		self.l = 1.0

		self.render_size = 50
		self.metadata['render.modes'] = ['rgb_array']

		# The state consists of the x,y coordinates of the shoulder, elbow and hand of the robot arm
		# and the (normalised) Cartesian velocity vectors of the shoulder and elbow
		self.observation_space = spaces.Box(-2,2, shape=(10,), dtype=np.float32)

		# Action is torque 
		if n_actions is None:
			self.action_space = spaces.Box(-self.max_torque,self.max_torque, (2,), np.float32)
			self.n_actions = n_actions
		else:
			self.n_actions = n_actions**2
			self.action_space = spaces.Discrete(self.n_actions)
			self.action_to_torque = dict()
			i = 0
			for tx in np.linspace(-self.max_torque, self.max_torque, num=n_actions):
				for ty in np.linspace(-self.max_torque, self.max_torque, num=n_actions):
					self.action_to_torque[i] = np.array([tx, ty])
					i += 1
			

	def update_state(self, hand_loc, elbow_loc, shoulder_loc, ang_vel_elbow, ang_vel_shoulder, torque):
		# First rotate elbow joint
		# move to coordinate system where elbow is the center
		shifted_elbow_loc = np.array([0., 0.])
		shifted_hand_loc = hand_loc - elbow_loc
		# update angular velocity of the elbow by dividing torque by moment of inertia
		ang_vel_elbow += self.dt * torque[1] / (self.m * self.l **2)
		ang_vel_elbow = np.clip(ang_vel_elbow, -self.max_speed, self.max_speed)
		del_angle = (self.dt * ang_vel_elbow) * 180. / np.pi
		shifted_hand_loc = np.dot(self._rotation_matrix(del_angle), shifted_hand_loc)
		# move to back original coordinate system
		hand_loc = shifted_hand_loc + elbow_loc

		# Rotate shoulder joint
		# move to coordinate system where shoulder is the center
		shifted_shoulder_loc = np.array([0., 0.])
		shifted_elbow_loc = elbow_loc - shoulder_loc
		shifted_hand_loc = hand_loc - shoulder_loc
		# update angular velocity of the shoulder by dividing torque by moment of inertia
		ang_vel_shoulder += self.dt * torque[0] / (self.m * self.l **2)
		ang_vel_shoulder = np.clip(ang_vel_shoulder, -self.max_speed, self.max_speed)
		del_angle = (self.dt * ang_vel_shoulder) * 180. / np.pi
		# rotate elbow and hand del_angle degrees
		shifted_elbow_loc = np.dot(self._rotation_matrix(del_angle), shifted_elbow_loc)
		shifted_hand_loc = np.dot(self._rotation_matrix(del_angle), shifted_hand_loc)
		# move to back original coordinate system
		elbow_loc = shifted_elbow_loc + shoulder_loc
		hand_loc = shifted_hand_loc + shoulder_loc

		return hand_loc, elbow_loc, shoulder_loc, ang_vel_elbow, ang_vel_shoulder
	
	def get_state(self, hand_loc, elbow_loc, shoulder_loc, ang_vel_elbow, ang_vel_shoulder):
		# transform the angular velocity into a velocity vector in Euclidean space
		shifted_elbow_loc = elbow_loc - shoulder_loc
		vel_elbow = np.cross(np.array([0,0,ang_vel_shoulder]), np.array([shifted_elbow_loc[0], shifted_elbow_loc[1], 0]))[:2]
		vel_elbow = 4 * vel_elbow / self.max_speed
		shifted_hand_loc = hand_loc - elbow_loc
		vel_hand = np.cross(np.array([0,0,ang_vel_elbow]), np.array([shifted_hand_loc[0], shifted_hand_loc[1], 0]))[:2]
		vel_hand = 4 * vel_hand / self.max_speed
		#vel_elbow = 4 * vel_elbow / self.max_speed

		return vel_elbow, vel_hand



	def step(self, action):
		previous_hand_loc = self.hand_loc

		# model the arm as two independent simple pendulums with zero gravity 
		# (this is not physically accurate, but much simpler to code)
		if self.n_actions is None:
			torque = np.clip(action, -self.max_torque, self.max_torque)
		else:
			if isinstance(action, np.ndarray):
				assert len(action.shape) == 1
				assert action.shape[0] == 1
				action = action[0]
			torque = self.action_to_torque[action]

		self.hand_loc, self.elbow_loc, self.shoulder_loc, self.ang_vel_elbow, self.ang_vel_shoulder = self.update_state(
			self.hand_loc, self.elbow_loc, self.shoulder_loc, self.ang_vel_elbow, self.ang_vel_shoulder, torque
		)
		self.base_hand_loc, self.base_elbow_loc, self.base_shoulder_loc, self.base_ang_vel_elbow, self.base_ang_vel_shoulder = self.update_state(
			self.base_hand_loc, self.base_elbow_loc, self.base_shoulder_loc, self.base_ang_vel_elbow, self.base_ang_vel_shoulder, torque
		)

		# reward 1 if hand is close enough to target locations or has been in between the current and last timestep
		terminated = False
		distance_to_target = segment_distance(self._target_location, previous_hand_loc, self.hand_loc)[0]
		if distance_to_target < self._epsilon:
			terminated = True
			reward = 1.0
		elif self.simple_r_function is False and distance_to_target < self.smallest_segment_dist:
			reward = (1. - (distance_to_target / 2.)) / (TIMEOUT_STEPS / 2.)
			self.smallest_segment_dist = distance_to_target
		else:
			reward = 0.0
			# reward = -(distance_to_target / 2.) / (TIMEOUT_STEPS / 2.)
		self.ep_rewards += reward

		self.step_counter += 1

		truncated = False
		if self.step_counter >= TIMEOUT_STEPS:
			truncated = True
		done = terminated or truncated

		info = {"TimeLimit.truncated": True} if truncated else {}
		info["level_seed"] = self._current_task_id
		if done:
			info["episode"] = {'r': self.ep_rewards}

		vel_elbow, vel_hand = self.get_state(self.hand_loc, self.elbow_loc, self.shoulder_loc, self.ang_vel_elbow, self.ang_vel_shoulder)


		# obs = np.array([*self.shoulder_loc, *self.elbow_loc, *self.hand_loc, 2*self.ang_vel_shoulder/self.max_speed, 2*self.ang_vel_elbow/self.max_speed], dtype=np.float32)
		obs = np.array([*self.shoulder_loc, *self.elbow_loc, *self.hand_loc, *vel_elbow, *vel_hand], dtype=np.float32)

		return obs, reward, done, info

	def get_base_state(self):
		vel_elbow, vel_hand = self.get_state(self.base_hand_loc, self.base_elbow_loc, self.base_shoulder_loc, self.base_ang_vel_elbow, self.base_ang_vel_shoulder)
			
		return np.array([*self.base_shoulder_loc, *self.base_elbow_loc, *self.base_hand_loc, *vel_elbow, *vel_hand], dtype=np.float32)


	def _rotation_matrix(self, angle):
		theta = (angle/180.) * np.pi
		return np.array([[np.cos(theta), -np.sin(theta)], 
						 [np.sin(theta),  np.cos(theta)]])

	def reset(self):
		if self.num_tasks > 0:
			self._current_task_id = (self._current_task_id + 1) % self.num_tasks

		self.shoulder_loc = np.array([0,1], dtype=np.float32)

		elbow_vector = np.array([0,-.5], dtype=np.float32)
		if self.num_tasks > 0:
			shoulder_angle = self.tasks[self._current_task_id][0]
		else:
			# shoulder_angle = np.random.randint(-45, 45)
			shoulder_angle = np.random.randint(0, 360)
		elbow_vector = np.dot(self._rotation_matrix(shoulder_angle), elbow_vector)
		self.elbow_loc = self.shoulder_loc + elbow_vector

		hand_vector = .5 * (-1. * self.elbow_loc) / np.linalg.norm(self.elbow_loc)
		if self.num_tasks > 0:
			elbow_angle = self.tasks[self._current_task_id][1]
		else:
			# elbow_angle = np.random.randint(-90, 90)
			elbow_angle = np.random.randint(0, 360)
		hand_vector = np.dot(self._rotation_matrix(elbow_angle), hand_vector)
		self.hand_loc = self.elbow_loc + hand_vector

		# rotate the entire arm 
		if self.num_tasks > 0:
			angle_along_unit_circle = self.tasks[self._current_task_id][2]
		else:
			if isinstance(self.tasks, int):
				angle_along_unit_circle = self.tasks
			else:
				angle_along_unit_circle = np.random.randint(0, 360)
		self.base_shoulder_loc = self.shoulder_loc.copy()
		self.base_elbow_loc = self.elbow_loc.copy()
		self.base_hand_loc = self.hand_loc.copy()
		self.shoulder_loc = np.dot(self._rotation_matrix(angle_along_unit_circle), self.shoulder_loc)
		self.elbow_loc = np.dot(self._rotation_matrix(angle_along_unit_circle), self.elbow_loc)
		self.hand_loc = np.dot(self._rotation_matrix(angle_along_unit_circle), self.hand_loc)

		self.ang_vel_shoulder = 0.0
		self.ang_vel_elbow = 0.0
		self.base_ang_vel_shoulder = self.ang_vel_shoulder
		self.base_ang_vel_elbow = self.ang_vel_elbow

		self.step_counter = 0
		self.ep_rewards = 0
		self.smallest_segment_dist = np.linalg.norm(self.hand_loc)
		
		return np.array([*self.shoulder_loc, *self.elbow_loc, *self.hand_loc, self.ang_vel_shoulder, self.ang_vel_shoulder, self.ang_vel_elbow, self.ang_vel_elbow], dtype=np.float32)
	
	def _loc_to_pixel(self, location):
		scaled_location = location * (self.render_size / 2) * np.array([1., -1.])
		shifted_location = scaled_location + np.array([self.render_size / 2, self.render_size / 2])
		int_location = np.round(shifted_location).astype(np.int32)

		return np.clip(int_location + np.array([self.render_size // 2, self.render_size // 2]), 0, 2*self.render_size)

	def render(self, mode=None):
		img = np.ones((3,2*self.render_size, 2*self.render_size))

		# # Paint target region black
		# for loc in self._target_region:
		#     loc_inds = self._loc_to_pixel(loc)
		#     img[:, loc_inds[1], loc_inds[0]] = np.array([0.,0.,0.])
		# Paint target location black
		goal_inds = self._loc_to_pixel(self._target_location)
		img[:, goal_inds[1], goal_inds[0]] = np.array([0.,0.,0.])

		# Paint shoulder red
		shoulder_inds = self._loc_to_pixel(self.shoulder_loc)
		img[:, shoulder_inds[1], shoulder_inds[0]] = np.array([1.,0.,0.])
		# self._paint_region_around(img, shoulder_inds, np.array([1.,0.,0.]))

		# Paint elbow green
		elbow_inds = self._loc_to_pixel(self.elbow_loc)
		img[:, elbow_inds[1], elbow_inds[0]] = np.array([0.,1.,0.])
		# self._paint_region_around(img, elbow_inds, np.array([0.,1.,0.]))

		# Paint hand blue
		hand_inds = self._loc_to_pixel(self.hand_loc)
		img[:, hand_inds[1], hand_inds[0]] = np.array([0.,0.,1.])
		# self._paint_region_around(img, hand_inds, np.array([0.,0.,1.]))

		return np.transpose(img, (1,2,0))
	
	def _paint_region_around(self, img, inds, color):
		img[:, inds[1]-1, inds[0]] = color
		img[:, inds[1]-1, inds[0]-1] = color
		img[:, inds[1]-1, inds[0]+1] = color
		img[:, inds[1]+1, inds[0]] = color
		img[:, inds[1]+1, inds[0]-1] = color
		img[:, inds[1]+1, inds[0]+1] = color
		img[:, inds[1], inds[0]-1] = color
		img[:, inds[1], inds[0]+1] = color

class ControlIllustrativeVenv(gym.Env):
	def __init__(self, n_envs=1, device='cpu', tasks=None, g=10.0, n_actions=None, simple_r_function=False, **kwargs):
		self.envs = [ControlIllustrativeCMDP(tasks=tasks, g=g, n_actions=n_actions, simple_r_function=simple_r_function, **kwargs) for _ in range(n_envs)]
		self.n_envs = n_envs
		self.observation_space = self.envs[0].observation_space
		self.action_space = self.envs[0].action_space
		self.device=device

	def step(self, action):
		obs_n = []
		reward_n = []
		done_n = []
		info_n = []
		for i, a in enumerate(action):
			obs, reward, done, info = self.envs[i].step(a.cpu().numpy())
			if done:
				obs = self.envs[i].reset()
			obs_n.append(obs)
			reward_n.append(reward)
			done_n.append(done)
			info_n.append(info)

		obs = torch.as_tensor(np.stack(obs_n), device=self.device)
		rewards = torch.as_tensor(np.array(reward_n)[:, np.newaxis], device=self.device)
		dones = torch.as_tensor(np.array(done_n, dtype=np.bool_), device=self.device)
		infos = info_n

		return obs, rewards, dones, infos
	
	def reset(self):
		obs_n = []
		for i in range(self.n_envs):
			obs = self.envs[i].reset()
			obs_n.append(obs)
		return torch.as_tensor(np.stack(obs_n), device=self.device)
		