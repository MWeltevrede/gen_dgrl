import numpy as np

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
    def __init__(self, tasks=[(45,-45,0), (45,-45,90), (45,-45,180), (45,-45,270)], g=10.0, **kwargs):
        self.tasks = tasks
        self._current_task_id = 0
        self.num_tasks = len(tasks)
        self._target_location = np.array([0.0,0.0])
        self._epsilon = 0.01
        self.step_counter = 0

        self.max_speed = 4
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = g
        self.m = 1.0
        self.l = 1.0

        self.render_size = 50

        # The state consists of the x,y coordinates of the shoulder, elbow and hand of the robot arm
        # and the (normalised) Cartesian velocity vectors of the shoulder and elbow
        self.observation_space = spaces.Box(-2,2, shape=(10,), dtype=np.float32)

        # Action is torque 
        self.action_space = spaces.Box(-self.max_torque,self.max_torque, (2,), np.float32)

    def step(self, action):
        previous_hand_loc = self.hand_loc

        # model the arm as two independent simple pendulums with zero gravity 
        # (this is not physically accurate, but much simpler to code)
        torque = np.clip(action, -self.max_torque, self.max_torque)

        # First rotate elbow joint
        # move to coordinate system where elbow is the center
        shifted_elbow_loc = np.array([0., 0.])
        shifted_hand_loc = self.hand_loc - self.elbow_loc
        # update angular velocity of the elbow by dividing torque by moment of inertia
        self.ang_vel_elbow += self.dt * torque[1] / (self.m * self.l **2)
        self.ang_vel_elbow = np.clip(self.ang_vel_elbow, -self.max_speed, self.max_speed)
        del_angle = (self.dt * self.ang_vel_elbow) * 180. / np.pi
        shifted_hand_loc = np.dot(self._rotation_matrix(del_angle), shifted_hand_loc)
        # move to back original coordinate system
        self.hand_loc = shifted_hand_loc + self.elbow_loc

        # Rotate shoulder joint
        # move to coordinate system where shoulder is the center
        shifted_shoulder_loc = np.array([0., 0.])
        shifted_elbow_loc = self.elbow_loc - self.shoulder_loc
        shifted_hand_loc = self.hand_loc - self.shoulder_loc
        # update angular velocity of the shoulder by dividing torque by moment of inertia
        self.ang_vel_shoulder += self.dt * torque[0] / (self.m * self.l **2)
        self.ang_vel_shoulder = np.clip(self.ang_vel_shoulder, -self.max_speed, self.max_speed)
        del_angle = (self.dt * self.ang_vel_shoulder) * 180. / np.pi
        # rotate elbow and hand del_angle degrees
        shifted_elbow_loc = np.dot(self._rotation_matrix(del_angle), shifted_elbow_loc)
        shifted_hand_loc = np.dot(self._rotation_matrix(del_angle), shifted_hand_loc)
        # move to back original coordinate system
        self.elbow_loc = shifted_elbow_loc + self.shoulder_loc
        self.hand_loc = shifted_hand_loc + self.shoulder_loc

        # reward 1 if hand is close enough to target locations or has been in between the current and last timestep
        reward = 0
        if segment_distance(self._target_location, previous_hand_loc, self.hand_loc) < self._epsilon:
            reward = 1

        self.step_counter += 1

        terminated = reward == 1
        truncated = False
        if self.step_counter >= TIMEOUT_STEPS:
            truncated = True
        done = terminated or truncated

        info = {"TimeLimit.truncated": True} if truncated else {}
        info["level_seed"] = self._current_task_id

        # transform the angular velocity into a velocity vector in Euclidean space
        vel_elbow = np.cross(np.array([0,0,self.ang_vel_shoulder]), np.array([self.elbow_loc[0], self.elbow_loc[1], 0]))[:2]
        vel_elbow = 4 * vel_elbow / self.max_speed
        vel_hand = np.cross(np.array([0,0,self.ang_vel_elbow]), np.array([self.hand_loc[0], self.hand_loc[1], 0]))[:2]
        vel_elbow = 4 * vel_elbow / self.max_speed


        # obs = np.array([*self.shoulder_loc, *self.elbow_loc, *self.hand_loc, 2*self.ang_vel_shoulder/self.max_speed, 2*self.ang_vel_elbow/self.max_speed], dtype=np.float32)
        obs = np.array([*self.shoulder_loc, *self.elbow_loc, *self.hand_loc, *vel_elbow, *vel_hand], dtype=np.float32)

        return obs, reward, done, info



    def _rotation_matrix(self, angle):
        theta = (angle/180.) * np.pi
        return np.array([[np.cos(theta), -np.sin(theta)], 
                         [np.sin(theta),  np.cos(theta)]])

    def reset(self):
        self._current_task_id = (self._current_task_id + 1) % self.num_tasks

        self.shoulder_loc = np.array([0,1], dtype=np.float32)

        elbow_vector = np.array([0,-.5], dtype=np.float32)
        shoulder_angle = self.tasks[self._current_task_id][0]
        elbow_vector = np.dot(self._rotation_matrix(shoulder_angle), elbow_vector)
        self.elbow_loc = self.shoulder_loc + elbow_vector

        hand_vector = .5 * (-1. * self.elbow_loc) / np.linalg.norm(self.elbow_loc)
        elbow_angle = self.tasks[self._current_task_id][1]
        hand_vector = np.dot(self._rotation_matrix(elbow_angle), hand_vector)
        self.hand_loc = self.elbow_loc + hand_vector

        # rotate the entire arm 
        angle_along_unit_circle = self.tasks[self._current_task_id][2]
        self.shoulder_loc = np.dot(self._rotation_matrix(angle_along_unit_circle), self.shoulder_loc)
        self.elbow_loc = np.dot(self._rotation_matrix(angle_along_unit_circle), self.elbow_loc)
        self.hand_loc = np.dot(self._rotation_matrix(angle_along_unit_circle), self.hand_loc)

        self.ang_vel_shoulder = 0.0
        self.ang_vel_elbow = 0.0

        self.step_counter = 0
        
        return np.array([*self.shoulder_loc, *self.elbow_loc, *self.hand_loc, self.ang_vel_shoulder, self.ang_vel_shoulder, self.ang_vel_elbow, self.ang_vel_elbow], dtype=np.float32)
    
    def _loc_to_pixel(self, location):
        scaled_location = location * (self.render_size / 2) * np.array([1., -1.])
        shifted_location = scaled_location + np.array([self.render_size / 2, self.render_size / 2])
        int_location = np.round(shifted_location).astype(np.int32)

        return np.clip(int_location + np.array([self.render_size // 2, self.render_size // 2]), 0, 2*self.render_size)

    def render(self):
        img = np.ones((3,2*self.render_size, 2*self.render_size))

        # Paint target location black
        goal_inds = self._loc_to_pixel(self._target_location)
        img[:, goal_inds[1], goal_inds[0]] = np.array([0.,0.,0.])

        # Paint shoulder red
        shoulder_inds = self._loc_to_pixel(self.shoulder_loc)
        img[:, shoulder_inds[1], shoulder_inds[0]] = np.array([1.,0.,0.])

        # Paint elbow green
        elbow_inds = self._loc_to_pixel(self.elbow_loc)
        img[:, elbow_inds[1], elbow_inds[0]] = np.array([0.,1.,0.])

        # Paint hand blue
        hand_inds = self._loc_to_pixel(self.hand_loc)
        img[:, hand_inds[1], hand_inds[0]] = np.array([0.,0.,1.])

        return img
