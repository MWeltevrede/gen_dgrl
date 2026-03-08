import numpy as np
import torch as th
import math
import kornia 

def identity(batch):
	return batch
	
def crop(batch):
	aug_trans = th.nn.Sequential(th.nn.ReplicationPad2d(12),
							kornia.augmentation.RandomCrop((64, 64)))
	return aug_trans(batch)

def _rotation_matrix(angle, device):
	theta = th.tensor((angle/180.) * math.pi, device=device)
	return th.tensor([[th.cos(theta), -th.sin(theta)], 
						[th.sin(theta),  th.cos(theta)]], device=device)	# [2,2]

C4_MATRICES = [
	_rotation_matrix(0, th.device('cuda')), 
	_rotation_matrix(90, th.device('cuda')), 
	_rotation_matrix(180, th.device('cuda')), 
	_rotation_matrix(270, th.device('cuda')), 
]

C90_MATRICES = [_rotation_matrix(angle, th.device('cuda')) for angle in range(0, 360, 4)]

def rotate(batch, angles):
	if len(batch.shape) == 1:
		batch = batch.unsqueeze(0)
		angles = [angles]
	shoulder_loc = batch[:, :2]		# [batch_size, 2]
	elbow_loc = batch[:, 2:4]
	hand_loc = batch[:, 4:6]
	vel_elbow = batch[:, 6:8]
	vel_hand = batch[:, 8:]

	# rotate state by angle
	vectorized_batch = th.stack([shoulder_loc, elbow_loc, hand_loc, vel_elbow, vel_hand], dim=0)	# [5, batch_size, 2]
	rotated_vectorized_batch = th.empty_like(vectorized_batch)
	for angle in [0, 90, 180, 270]:
		inds_with_angle = np.where(np.array(angles) == angle)[0]
		rotation_matrix = C4_MATRICES[angle // 90].T.unsqueeze(0).expand(5, -1, -1)		# [5, 2, 2]
		rotated_vectorized_batch[:, inds_with_angle] = th.bmm(vectorized_batch[:, inds_with_angle], rotation_matrix)		# [5, inds_with_angle_size, 2]


	return th.concatenate([rotated_vectorized_batch[0], rotated_vectorized_batch[1], rotated_vectorized_batch[2], rotated_vectorized_batch[3], rotated_vectorized_batch[4]], dim=-1)


def rotate_c90(obs):
	shoulder_loc = obs[:2]		# [batch_size, 2]
	elbow_loc = obs[2:4]
	hand_loc = obs[4:6]
	vel_elbow = obs[6:8]
	vel_hand = obs[8:]

	# rotate state by angle
	vectorized_batch = th.stack([shoulder_loc, elbow_loc, hand_loc, vel_elbow, vel_hand], dim=0).unsqueeze(1)	# [5, 1, 2]
	vectorized_batch = th.repeat_interleave(vectorized_batch, 90, dim=1)	# [5, 90, 2]
	rotated_vectorized_batch = th.empty_like(vectorized_batch)
	for i, rot_mat in enumerate(C90_MATRICES):
		inds_with_angle = [i]
		rotation_matrix = rot_mat.T.unsqueeze(0).expand(5, -1, -1)		# [5, 2, 2]
		rotated_vectorized_batch[:, inds_with_angle] = th.bmm(vectorized_batch[:, inds_with_angle], rotation_matrix)		# [5, inds_with_angle_size, 2]


	return th.concatenate([rotated_vectorized_batch[0], rotated_vectorized_batch[1], rotated_vectorized_batch[2], rotated_vectorized_batch[3], rotated_vectorized_batch[4]], dim=-1)

