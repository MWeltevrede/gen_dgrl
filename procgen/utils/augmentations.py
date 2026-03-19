import numpy as np
import torch as th
import math
import kornia 
import random

def identity(batch):
	return batch
	
def crop(batch):
	aug_trans = th.nn.Sequential(th.nn.ReplicationPad2d(12),
							kornia.augmentation.RandomCrop((64, 64)))
	return aug_trans(batch)

		
def random_conv(x):
	x = x / 255.
	_device = x.device
	
	img_h, img_w = x.shape[2], x.shape[3]
	num_stack_channel = x.shape[1]
	num_batch = x.shape[0]
	num_trans = num_batch
	batch_size = int(num_batch / num_trans)

	with th.no_grad():
		# initialize random covolution
		rand_conv = th.nn.Conv2d(3, 3, kernel_size=3, bias=False, padding=1).to(_device)
		
		for trans_index in range(num_trans):
			th.nn.init.xavier_normal_(rand_conv.weight.data)
			temp_x = x[trans_index*batch_size:(trans_index+1)*batch_size]
			temp_x = temp_x.reshape(-1, 3, img_h, img_w) # (batch x stack, channel, h, w)
			rand_out = rand_conv(temp_x)
			if trans_index == 0:
				total_out = rand_out
			else:
				total_out = th.cat((total_out, rand_out), 0)
		total_out = total_out.reshape(-1, num_stack_channel, img_h, img_w)
	return total_out * 255.
		
		
def color_jitter(inputs):
	inputs = inputs / 255.
	brightness = [0.8, 1.2]
	contrast = [0.8, 1.2]
	saturation = [0.8, 1.2]
	hue = [-0.25, 0.25]
	prob = 1
	_device = inputs.device

	# batch size
	random_inds = np.random.choice(
		[True, False], len(inputs), p=[prob, 1 - prob])
	inds = th.tensor(random_inds).to(_device)
	if random_inds.sum() > 0:	
		# Shuffle transform
		if random.uniform(0,1) >= 0.5:
			inputs[inds] = adjust_contrast(inputs[inds], contrast)
			inputs[inds] = rgb2hsv(inputs[inds])
			inputs[inds] = adjust_brightness(inputs[inds], brightness)
			inputs[inds] = adjust_hue(inputs[inds], hue)
			inputs[inds] = adjust_saturate(inputs[inds], saturation)
			inputs[inds] = hsv2rgb(inputs[inds])
		else:
			inputs[inds] = rgb2hsv(inputs[inds])
			inputs[inds] = adjust_brightness(inputs[inds], brightness)
			inputs[inds] = adjust_hue(inputs[inds], hue)
			inputs[inds] = adjust_saturate(inputs[inds], saturation)
			inputs[inds] = hsv2rgb(inputs[inds])
			inputs[inds] = adjust_contrast(inputs[inds], contrast)

	return inputs * 255.

def adjust_contrast(x, contrast):
	factor_contrast = th.empty(x.shape[0], device=x.device).uniform_(*contrast)
	factor_contrast = factor_contrast.reshape(-1,1).repeat(1, 1).reshape(-1)
	
	means = th.mean(x, dim=(2, 3), keepdim=True)
	return th.clamp((x - means)
						* factor_contrast.view(len(x), 1, 1, 1) + means, 0, 1)

def adjust_hue(x, hue):
	factor_hue = th.empty(x.shape[0], device=x.device).uniform_(*hue)
	factor_hue = factor_hue.reshape(-1,1).repeat(1, 1).reshape(-1)
	
	h = x[:, 0, :, :]
	h = h + (factor_hue.view(len(x), 1, 1) * 255. / 360.)
	h = (h % 1)
	x[:, 0, :, :] = h
	return x

def adjust_saturate(x, saturation):
	factor_saturate = th.empty(x.shape[0], device=x.device).uniform_(*saturation)
	factor_saturate = factor_saturate.reshape(-1,1).repeat(1, 1).reshape(-1)
	
	x[:, 1, :, :] = th.clamp(x[:, 1, :, :]
								* factor_saturate.view(len(x), 1, 1), 0, 1)
	return th.clamp(x, 0, 1)

def adjust_brightness(x, brightness):
	factor_brightness = th.empty(x.shape[0], device=x.device).uniform_(*brightness)
	factor_brightness = factor_brightness.reshape(-1,1).repeat(1, 1).reshape(-1)
	
	x[:, 2, :, :] = th.clamp(x[:, 2, :, :]
									* factor_brightness.view(len(x), 1, 1), 0, 1)
	return th.clamp(x, 0, 1)

def rgb2hsv(rgb, eps=1e-8):
	# Reference: https://www.rapidtables.com/convert/color/rgb-to-hsv.html
	# Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

	_device = rgb.device
	r, g, b = rgb[:, 0, :, :], rgb[:, 1, :, :], rgb[:, 2, :, :]

	Cmax = rgb.max(1)[0]
	Cmin = rgb.min(1)[0]
	delta = Cmax - Cmin

	hue = th.zeros((rgb.shape[0], rgb.shape[2], rgb.shape[3])).to(_device)
	hue[Cmax== r] = (((g - b)/(delta + eps)) % 6)[Cmax == r]
	hue[Cmax == g] = ((b - r)/(delta + eps) + 2)[Cmax == g]
	hue[Cmax == b] = ((r - g)/(delta + eps) + 4)[Cmax == b]
	hue[Cmax == 0] = 0.0
	hue = hue / 6. # making hue range as [0, 1.0)
	hue = hue.unsqueeze(dim=1)

	saturation = (delta) / (Cmax + eps)
	saturation[Cmax == 0.] = 0.
	saturation = saturation.to(_device)
	saturation = saturation.unsqueeze(dim=1)

	value = Cmax
	value = value.to(_device)
	value = value.unsqueeze(dim=1)

	return th.cat((hue, saturation, value), dim=1)

def hsv2rgb(hsv):
	# Reference: https://www.rapidtables.com/convert/color/hsv-to-rgb.html
	# Reference: https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L287

	_device = hsv.device

	hsv = th.clamp(hsv, 0, 1)
	hue = hsv[:, 0, :, :] * 360.
	saturation = hsv[:, 1, :, :]
	value = hsv[:, 2, :, :]

	c = value * saturation
	x = - c * (th.abs((hue / 60.) % 2 - 1) - 1)
	m = (value - c).unsqueeze(dim=1)

	rgb_prime = th.zeros_like(hsv).to(_device)

	inds = (hue < 60) * (hue >= 0)
	rgb_prime[:, 0, :, :][inds] = c[inds]
	rgb_prime[:, 1, :, :][inds] = x[inds]

	inds = (hue < 120) * (hue >= 60)
	rgb_prime[:, 0, :, :][inds] = x[inds]
	rgb_prime[:, 1, :, :][inds] = c[inds]

	inds = (hue < 180) * (hue >= 120)
	rgb_prime[:, 1, :, :][inds] = c[inds]
	rgb_prime[:, 2, :, :][inds] = x[inds]

	inds = (hue < 240) * (hue >= 180)
	rgb_prime[:, 1, :, :][inds] = x[inds]
	rgb_prime[:, 2, :, :][inds] = c[inds]

	inds = (hue < 300) * (hue >= 240)
	rgb_prime[:, 2, :, :][inds] = c[inds]
	rgb_prime[:, 0, :, :][inds] = x[inds]

	inds = (hue < 360) * (hue >= 300)
	rgb_prime[:, 2, :, :][inds] = x[inds]
	rgb_prime[:, 0, :, :][inds] = c[inds]

	rgb = rgb_prime + th.cat((m, m, m), dim=1)
	rgb = rgb.to(_device)

	return th.clamp(rgb, 0, 1)


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

