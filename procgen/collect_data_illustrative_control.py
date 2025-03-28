from gym.envs.registration import register
import gym
import numpy as np
import io, os
import torch
import copy
import json

from stable_baselines3 import SAC

def generate_dataset(tasks, dataset_name, model):
    env = gym.make('ControlIllustrativeCMDP-v0', tasks=tasks)
    env.seed(88)
    obs = env.reset()

    dataset_dirname = f'datasets/{dataset_name}'
    os.makedirs(dataset_dirname, exist_ok=True)

    ep_obs = [obs]
    ep_rewards = []
    ep_dones = []
    ep_actions = []
    total_steps = 0
    for i in range(len(tasks)):
        done = False
        step = 0
        while not done:
            obs = torch.from_numpy(obs).float().to('cuda')
            if len(obs.shape) == 1:
                # add batch dimension
                obs = obs.unsqueeze(0)
            with torch.no_grad():
                action = model.policy._predict(obs, deterministic=True)
            # using numpy, if action is of shape [1,1] then convert it to [1]
            action = action.cpu().numpy()
            if len(action.shape) == 2:
                action = action[0]

            # action = optimal_policy(step)
            obs, reward, done, _ = env.step(np.array(action))
            ep_obs.append(obs)
            ep_rewards.append([reward])
            ep_dones.append(done)
            ep_actions.append(action)
            step += 1

            if done:
                total_steps += step
                episode = {'observations': np.array(ep_obs), 'actions': np.array(ep_actions), 'rewards': np.array(ep_rewards), 'dones': np.array(ep_dones)}
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
        i += 1

    print(total_steps)

#### Fixed Pose
# # C4 Rotations
# train_tasks = [(45,-45,0), (45,-45,90), (45,-45,180), (45,-45,270)]

# # Random Rotations
# train_tasks = [(45,-45,-6), (45,-45,94), (45,-45,171), (45,-45,289)]

# test_tasks = [(45,-45,-11), (45,-45,11), (45,-45,65), (45,-45,99), (45,-45,167), (45,-45,204), (45,-45,259), (45,-45,325)]


#### Random Pose
# # Base tasks: same rotations as the random rotations above, but with randomised starting poses for the robot arm
# train_tasks = [
#     (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
# ]

# # Now we perform 'data augmentation' by enumerating all poses for all rotations for the base set of tasks above
# train_tasks = [
#     (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
#     (105,213,-6), (152,290,94), (358,43,171), (248,77,289),     # duplicate all the poses in all tasks
#     (152,290,-6), (358,43,94), (248,77,171), (105,213,289),     # duplicate all the poses in all tasks
#     (358,43,-6), (248,77,94), (105,213,171), (152,290,289),     # duplicate all the poses in all tasks
# ]

# # Now we imitate random pure exploration at the start of each episode (ExploreGo)
# # by adding three random poses to each of the base tasks
# train_tasks = [
#     (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
#     (288,75,-6), (213,17,94), (56,215,171), (357,274,289),     # random poses in all tasks
#     (40,86,-6), (305,136,94), (187,91,171), (101,240,289),     # random poses in all tasks
#     (129,254,-6), (117,352,94), (60,167,171), (258,214,289),     # random poses in all tasks
# ]

# # Now we add completely new (unreachable) tasks by randomly sampling from the full distribution
# train_tasks = [
#     (248,77,-6), (105,213,94), (152,290,171), (358,43,289),     # unique poses
#     (288,75,20), (213,17,305), (56,215,269), (357,274,293),     # random poses and rotations in all tasks
#     (40,86,185), (305,136,2), (187,91,350), (101,240,101),     # random poses and rotations in all tasks
#     (129,254,22), (117,352,57), (60,167,118), (258,214,230),     # random poses and rotations in all tasks
#     (187,2,207), (337,91,143), (197,146 ,54), (138,293,200),     # random poses and rotations in all tasks
#     (38,195,284), (78,259,1), (97,139,19), (109,53,261),     # random poses and rotations in all tasks
#     (60,217,235), (39,252,162), (304,307,283), (264,216,240),     # random poses and rotations in all tasks
#     (52,191,186), (164,179,149), (80,13,106), (124,95,265),     # random poses and rotations in all tasks
# ]

#### Randomly generated
set_id = 4
np.random.seed(set_id)

num_base_tasks = 8
base_tasks = np.random.randint(0, 360, (num_base_tasks, 3))

da_expl_tasks = []
for i in range(num_base_tasks):
    a = copy.deepcopy(base_tasks)
    a[:, -1] = np.roll(base_tasks[:, -1], i+1)
    da_expl_tasks.append(a)
da_expl_tasks = np.concatenate(da_expl_tasks, axis=0)

random_expl_tasks = []
for i in range(num_base_tasks):
    a = np.random.randint(0, 360, (num_base_tasks, 3))
    a[:, -1] = copy.deepcopy(base_tasks[:, -1])
    random_expl_tasks.append(a)
random_expl_tasks = np.concatenate(random_expl_tasks, axis=0)

unreachablex8_tasks = [copy.deepcopy(base_tasks)]
for i in range(num_base_tasks-1):
    a = np.random.randint(0, 360, (num_base_tasks, 3))
    unreachablex8_tasks.append(a)
unreachablex8_tasks = np.concatenate(unreachablex8_tasks, axis=0)

unreachablex4_tasks = [copy.deepcopy(base_tasks)]
for i in range(3):
    a = np.random.randint(0, 360, (num_base_tasks, 3))
    unreachablex4_tasks.append(a)
unreachablex4_tasks = np.concatenate(unreachablex4_tasks, axis=0)

unreachablex2_tasks = [copy.deepcopy(base_tasks)]
a = np.random.randint(0, 360, (num_base_tasks, 3))
unreachablex2_tasks.append(a)
unreachablex2_tasks = np.concatenate(unreachablex2_tasks, axis=0)

with open(f"datasets/task_sets_{set_id}.json", 'r') as file:
    task_sets_dict = json.load(file)
    assert task_sets_dict['base_tasks'] == base_tasks.tolist()
    assert task_sets_dict['da_expl_tasks'] == da_expl_tasks.tolist()
    assert task_sets_dict['random_expl_tasks'] == random_expl_tasks.tolist()
    assert task_sets_dict['unreachable_tasks'] == unreachablex8_tasks.tolist()
    del task_sets_dict['unreachable_tasks']
    task_sets_dict['unreachablex8_tasks'] = unreachablex8_tasks.tolist()
    task_sets_dict['unreachablex4_tasks'] = unreachablex4_tasks.tolist()
    task_sets_dict['unreachablex2_tasks'] = unreachablex2_tasks.tolist()

test_tasks = []     # full distribution of poses and rotations

register(
     id="ControlIllustrativeCMDP-v0",
     entry_point="control_illustrative_env:ControlIllustrativeCMDP",
)

def optimal_policy(step):
    # a handcrafted (basically) optimal policy
    if step < 12:
        return [-2, 2]
    else:
        return [2, 2]
    
model = SAC.load("sac_control_illustrative_full")

# generate_dataset(base_tasks, f'control_illustrative_base_{set_id}', model)
# generate_dataset(da_expl_tasks, f'control_illustrative_da_expl_{set_id}', model)
# generate_dataset(random_expl_tasks, f'control_illustrative_random_expl_{set_id}', model)
generate_dataset(unreachablex8_tasks, f'control_illustrative_unreachablex8_{set_id}', model)
generate_dataset(unreachablex4_tasks, f'control_illustrative_unreachablex4_{set_id}', model)
generate_dataset(unreachablex2_tasks, f'control_illustrative_unreachablex2_{set_id}', model)

# save tasks
with open(f"datasets/task_sets_{set_id}.json", 'w') as file:
    # json.dump({'base_tasks':base_tasks.tolist(), 'da_expl_tasks':da_expl_tasks.tolist(), 'random_expl_tasks':random_expl_tasks.tolist(), 'unreachable_tasks':unreachable_tasks.tolist()}, file)
    json.dump(task_sets_dict, file)
