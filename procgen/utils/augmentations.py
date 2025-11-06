import numpy as np
import torch as th
import math

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
#C4_MATRICES = [
#	_rotation_matrix(0, th.device('cpu')), 
#	_rotation_matrix(90, th.device('cpu')), 
#	_rotation_matrix(180, th.device('cpu')), 
#	_rotation_matrix(270, th.device('cpu')), 
#]

#def rotate(state, angle):
#	shoulder_loc = state[:2]
#	elbow_loc = state[2:4]
#	hand_loc = state[4:6]
#	vel_elbow = state[6:8]
#	vel_hand = state[8:]

#	# rotate state by angle
#	rotated_shoulder_loc = np.dot(_rotation_matrix(angle), shoulder_loc)
#	rotated_elbow_loc = np.dot(_rotation_matrix(angle), elbow_loc)
#	rotated_hand_loc = np.dot(_rotation_matrix(angle), hand_loc)
#	rotated_vel_elbow = np.dot(_rotation_matrix(angle), vel_elbow)
#	rotated_vel_hand = np.dot(_rotation_matrix(angle), vel_hand)

#	return np.array([*rotated_shoulder_loc, *rotated_elbow_loc, *rotated_hand_loc, *rotated_vel_elbow, *rotated_vel_hand], dtype=np.float32)

def rotate(batch, angles):
	if len(batch.shape) == 1:
		batch = batch.unsqueeze(0)
		angles = [angles]
	shoulder_loc = batch[:, :2]		# [batch_size, 2]
	elbow_loc = batch[:, 2:4]
	hand_loc = batch[:, 4:6]
	vel_elbow = batch[:, 6:8]
	vel_hand = batch[:, 8:]

	## rotate state by angle
	#rotated_shoulder_loc = []
	#rotated_elbow_loc = []
	#rotated_hand_loc = []
	#rotated_vel_elbow = []
	#rotated_vel_hand = []
	#for i, angle in enumerate(angles):
	#	rotated_shoulder_loc.append(th.matmul(_rotation_matrix(angle, batch.device), shoulder_loc[i])) 
	#	rotated_elbow_loc.append(th.matmul(_rotation_matrix(angle, batch.device), elbow_loc[i])) 
	#	rotated_hand_loc.append(th.matmul(_rotation_matrix(angle, batch.device), hand_loc[i])) 
	#	rotated_vel_elbow.append(th.matmul(_rotation_matrix(angle, batch.device), vel_elbow[i])) 
	#	rotated_vel_hand.append(th.matmul(_rotation_matrix(angle, batch.device), vel_hand[i])) 
	#rotated_shoulder_loc = th.stack(rotated_shoulder_loc, dim=0)
	#rotated_elbow_loc = th.stack(rotated_elbow_loc, dim=0)
	#rotated_hand_loc = th.stack(rotated_hand_loc, dim=0)
	#rotated_vel_elbow = th.stack(rotated_vel_elbow, dim=0)
	#rotated_vel_hand = th.stack(rotated_vel_hand, dim=0)

	# rotate state by angle
	vectorized_batch = th.stack([shoulder_loc, elbow_loc, hand_loc, vel_elbow, vel_hand], dim=0)	# [5, batch_size, 2]
	rotated_vectorized_batch = th.empty_like(vectorized_batch)
	for angle in [0, 90, 180, 270]:
		inds_with_angle = np.where(np.array(angles) == angle)[0]
		rotation_matrix = C4_MATRICES[angle // 90].T.unsqueeze(0).expand(5, -1, -1)		# [5, 2, 2]
		rotated_vectorized_batch[:, inds_with_angle] = th.bmm(vectorized_batch[:, inds_with_angle], rotation_matrix)		# [5, inds_with_angle_size, 2]


	return th.concatenate([rotated_vectorized_batch[0], rotated_vectorized_batch[1], rotated_vectorized_batch[2], rotated_vectorized_batch[3], rotated_vectorized_batch[4]], dim=-1)
	#return th.concatenate([rotated_shoulder_loc, rotated_elbow_loc, rotated_hand_loc, rotated_vel_elbow, rotated_vel_hand], dim=-1)

